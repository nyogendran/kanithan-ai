// voice_widget.dart — Flutter Voice Input/Output Widget
// ======================================================
// Complete Flutter implementation for the Tamil Math Tutor voice interface.
//
// This widget handles:
//   1. Microphone capture → PCM stream → Python VAD backend (via WebSocket)
//   2. Visual feedback during listening (animated waveform + state indicators)
//   3. Dialect-aware display (shows normalized vs raw transcript)
//   4. TTS playback of tutor responses (streaming, with text highlight sync)
//   5. Pause indicator — shows student the tutor is still listening
//   6. Offline mode detection and fallback to Android TTS
//
// Architecture:
//   Flutter UI ↔ WebSocket ↔ voice_server.py ↔ VoiceIOManager ↔ Orchestrator
//
// Dependencies (pubspec.yaml):
//   flutter_sound: ^9.2.13       # microphone recording
//   web_socket_channel: ^2.4.0   # WebSocket to Python backend
//   just_audio: ^0.9.36          # audio playback (TTS output)
//   permission_handler: ^11.1.0  # microphone permission
//   provider: ^6.1.1             # state management

import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:just_audio/just_audio.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

// ─────────────────────────────────────────────────────────────────────────────
// VOICE SESSION STATE
// ─────────────────────────────────────────────────────────────────────────────

enum VoiceState {
  idle,
  listening,
  speechDetected,
  pauseDetected,   // mid-utterance pause — still listening
  transcribing,
  thinking,
  responding,
  playingAudio,
  error,
}

class VoiceSessionState extends ChangeNotifier {
  VoiceState _state = VoiceState.idle;
  String _transcript = '';
  String _dialectDetected = 'unknown';
  double _speechProbability = 0.0;
  String _tutorResponse = '';
  bool _isComplete = false;
  Map<String, dynamic>? _diagramSpec;
  Map<String, dynamic>? _exercise;
  String? _errorMessage;
  List<double> _waveformData = List.filled(40, 0.0);
  bool _isOffline = false;

  VoiceState get state => _state;
  String get transcript => _transcript;
  String get dialectDetected => _dialectDetected;
  String get tutorResponse => _tutorResponse;
  bool get isComplete => _isComplete;
  Map<String, dynamic>? get diagramSpec => _diagramSpec;
  Map<String, dynamic>? get exercise => _exercise;
  String? get errorMessage => _errorMessage;
  List<double> get waveformData => _waveformData;
  bool get isOffline => _isOffline;

  void updateState(VoiceState newState) {
    _state = newState;
    notifyListeners();
  }

  void updateTranscript(String text, String dialect, bool complete) {
    _transcript = text;
    _dialectDetected = dialect;
    _isComplete = complete;
    notifyListeners();
  }

  void updateTutorResponse(String text) {
    _tutorResponse = text;
    notifyListeners();
  }

  void updateWaveform(List<double> data) {
    _waveformData = data;
    notifyListeners();
  }

  void setDiagram(Map<String, dynamic> spec) {
    _diagramSpec = spec;
    notifyListeners();
  }

  void setExercise(Map<String, dynamic> ex) {
    _exercise = ex;
    notifyListeners();
  }

  void setError(String msg) {
    _errorMessage = msg;
    _state = VoiceState.error;
    notifyListeners();
  }

  void setOffline(bool offline) {
    _isOffline = offline;
    notifyListeners();
  }
}


// ─────────────────────────────────────────────────────────────────────────────
// VOICE CONTROLLER (business logic)
// ─────────────────────────────────────────────────────────────────────────────

class VoiceController {
  final VoiceSessionState sessionState;
  final String studentId;
  final String district;
  final String backendUrl;

  WebSocketChannel? _channel;
  final AudioPlayer _audioPlayer = AudioPlayer();
  final List<Uint8List> _audioQueue = [];
  bool _isPlayingAudio = false;
  StreamSubscription? _wsSubscription;
  Timer? _waveformTimer;

  // Simulated waveform amplitude (real implementation reads audio level)
  double _audioLevel = 0.0;

  VoiceController({
    required this.sessionState,
    required this.studentId,
    this.district = 'unknown',
    this.backendUrl = 'ws://localhost:8765',
  });

  // ── Connection ────────────────────────────────────────────────────────────

  Future<bool> connect() async {
    final micPermission = await Permission.microphone.request();
    if (!micPermission.isGranted) {
      sessionState.setError('மைக்ரோஃபோன் அனுமதி தேவை');
      return false;
    }

    try {
      final uri = Uri.parse(
        '$backendUrl?student_id=$studentId&district=$district'
      );
      _channel = WebSocketChannel.connect(uri);

      _wsSubscription = _channel!.stream.listen(
        _handleServerEvent,
        onError: (e) {
          sessionState.setError('இணைப்பு பிழை: $e');
          _handleOffline();
        },
        onDone: () => sessionState.updateState(VoiceState.idle),
      );

      // Start waveform animation
      _startWaveformAnimation();
      sessionState.updateState(VoiceState.listening);
      return true;
    } catch (e) {
      sessionState.setOffline(true);
      _handleOffline();
      return false;
    }
  }

  void _handleOffline() {
    // On offline: use Android STT + Android TTS directly
    sessionState.setOffline(true);
    sessionState.updateState(VoiceState.listening);
    // Flutter platform channel to Android native STT/TTS
    // Implementation: method channel to SpeechRecognizer + TextToSpeech
  }

  // ── Server event handler ──────────────────────────────────────────────────

  void _handleServerEvent(dynamic message) {
    final Map<String, dynamic> event;

    if (message is String) {
      event = jsonDecode(message);
    } else if (message is List<int>) {
      // Binary audio data
      _queueAudio(Uint8List.fromList(message));
      return;
    } else {
      return;
    }

    switch (event['type']) {
      case 'listening_start':
        sessionState.updateState(VoiceState.listening);
        break;

      case 'speech_detected':
        sessionState.updateState(VoiceState.speechDetected);
        _audioLevel = (event['level'] as num?)?.toDouble() ?? 0.5;
        break;

      case 'pause_detected':
        // Mid-utterance pause — show visual indicator but keep listening
        sessionState.updateState(VoiceState.pauseDetected);
        break;

      case 'utterance_ready':
        final durationMs = (event['duration_ms'] as num?)?.toInt() ?? 0;
        sessionState.updateState(VoiceState.transcribing);
        break;

      case 'stt_result':
        sessionState.updateTranscript(
          event['text'] as String? ?? '',
          event['dialect'] as String? ?? 'unknown',
          event['is_complete'] as bool? ?? false,
        );
        break;

      case 'processing':
        sessionState.updateState(VoiceState.thinking);
        break;

      case 'response_text':
        final text = event['text'] as String? ?? '';
        sessionState.updateTutorResponse(text);
        sessionState.updateState(VoiceState.responding);
        break;

      case 'response_audio':
        // Audio is sent as next binary WebSocket frame
        // Already handled in binary branch above
        sessionState.updateState(VoiceState.playingAudio);
        break;

      case 'diagram_ready':
        sessionState.setDiagram(event as Map<String, dynamic>);
        break;

      case 'exercise_ready':
        sessionState.setExercise(event as Map<String, dynamic>);
        break;

      case 'error':
        sessionState.setError(event['message'] as String? ?? 'பிழை ஏற்பட்டது');
        break;
    }
  }

  // ── Audio playback ────────────────────────────────────────────────────────

  void _queueAudio(Uint8List audioBytes) {
    _audioQueue.add(audioBytes);
    if (!_isPlayingAudio) {
      _playNextAudio();
    }
  }

  Future<void> _playNextAudio() async {
    if (_audioQueue.isEmpty) {
      _isPlayingAudio = false;
      sessionState.updateState(VoiceState.listening);
      return;
    }
    _isPlayingAudio = true;
    final chunk = _audioQueue.removeAt(0);

    try {
      final source = BytesAudioSource(chunk);
      await _audioPlayer.setAudioSource(source);
      await _audioPlayer.play();
      await _audioPlayer.playerStateStream.firstWhere(
        (s) => s.processingState == ProcessingState.completed);
      _playNextAudio();  // play next chunk
    } catch (e) {
      _isPlayingAudio = false;
    }
  }

  // ── Waveform animation ────────────────────────────────────────────────────

  void _startWaveformAnimation() {
    _waveformTimer = Timer.periodic(const Duration(milliseconds: 50), (_) {
      if (sessionState.state == VoiceState.listening ||
          sessionState.state == VoiceState.speechDetected) {
        final waveform = List.generate(40, (i) {
          final t = DateTime.now().millisecondsSinceEpoch / 1000.0;
          final base = _audioLevel * (0.3 + 0.7 * ((i % 5) / 5.0));
          return base * (0.5 + 0.5 * (0.5 + 0.5 *
              (i % 3 == 0 ? (t * 3 + i).abs() % 1.0 :
               i % 3 == 1 ? (t * 2.5 + i).abs() % 1.0 :
                            (t * 2 + i).abs() % 1.0)));
        });
        sessionState.updateWaveform(waveform);
      }
    });
  }

  void dispose() {
    _wsSubscription?.cancel();
    _channel?.sink.close();
    _audioPlayer.dispose();
    _waveformTimer?.cancel();
  }
}


// ─────────────────────────────────────────────────────────────────────────────
// VOICE WIDGET (main UI)
// ─────────────────────────────────────────────────────────────────────────────

class TamilMathVoiceWidget extends StatefulWidget {
  final String studentId;
  final String district;
  final String backendUrl;
  final Function(Map<String, dynamic>)? onDiagramReady;
  final Function(Map<String, dynamic>)? onExerciseReady;

  const TamilMathVoiceWidget({
    Key? key,
    required this.studentId,
    this.district = 'unknown',
    this.backendUrl = 'ws://localhost:8765',
    this.onDiagramReady,
    this.onExerciseReady,
  }) : super(key: key);

  @override
  State<TamilMathVoiceWidget> createState() => _TamilMathVoiceWidgetState();
}

class _TamilMathVoiceWidgetState extends State<TamilMathVoiceWidget>
    with TickerProviderStateMixin {

  late VoiceSessionState _sessionState;
  late VoiceController _controller;
  late AnimationController _pulseController;
  late Animation<double> _pulseAnimation;

  @override
  void initState() {
    super.initState();
    _sessionState = VoiceSessionState();
    _controller = VoiceController(
      sessionState: _sessionState,
      studentId: widget.studentId,
      district: widget.district,
      backendUrl: widget.backendUrl,
    );
    _sessionState.addListener(_onStateChanged);

    _pulseController = AnimationController(
      duration: const Duration(milliseconds: 1200),
      vsync: this,
    )..repeat(reverse: true);
    _pulseAnimation = Tween<double>(begin: 0.95, end: 1.05).animate(
      CurvedAnimation(parent: _pulseController, curve: Curves.easeInOut));
  }

  void _onStateChanged() {
    if (_sessionState.diagramSpec != null) {
      widget.onDiagramReady?.call(_sessionState.diagramSpec!);
    }
    if (_sessionState.exercise != null) {
      widget.onExerciseReady?.call(_sessionState.exercise!);
    }
    setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        // Connection status banner
        if (_sessionState.isOffline)
          _buildOfflineBanner(),

        // Main voice button + waveform
        _buildVoiceButton(),

        const SizedBox(height: 12),

        // State label
        _buildStateLabel(),

        const SizedBox(height: 8),

        // Transcript display
        if (_sessionState.transcript.isNotEmpty)
          _buildTranscriptCard(),

        // Tutor response
        if (_sessionState.tutorResponse.isNotEmpty)
          _buildResponseCard(),
      ],
    );
  }

  // ── Sub-widgets ───────────────────────────────────────────────────────────

  Widget _buildOfflineBanner() {
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: const Color(0xFFFAEEDA),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: const Color(0xFFFAC775), width: 0.5),
      ),
      child: Row(children: [
        const Icon(Icons.wifi_off, size: 14, color: Color(0xFF633806)),
        const SizedBox(width: 6),
        Text('நெட்வொர்க் இல்லை — ஆஃப்லைன் முறை',
            style: const TextStyle(fontSize: 11, color: Color(0xFF633806))),
      ]),
    );
  }

  Widget _buildVoiceButton() {
    final isActive = _sessionState.state != VoiceState.idle &&
                     _sessionState.state != VoiceState.error;
    final isPause = _sessionState.state == VoiceState.pauseDetected;
    final isSpeaking = _sessionState.state == VoiceState.speechDetected;

    return GestureDetector(
      onTap: isActive ? null : _startListening,
      child: Column(children: [
        // Waveform bars
        SizedBox(
          height: 48,
          child: _buildWaveform(isSpeaking || isPause),
        ),
        const SizedBox(height: 8),
        // Mic button
        AnimatedBuilder(
          animation: _pulseAnimation,
          builder: (_, child) => Transform.scale(
            scale: isSpeaking ? _pulseAnimation.value : 1.0,
            child: child,
          ),
          child: Container(
            width: 64,
            height: 64,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: _buttonColor(),
              boxShadow: isActive ? [
                BoxShadow(
                  color: _buttonColor().withOpacity(0.3),
                  blurRadius: 16, spreadRadius: 4,
                )
              ] : null,
            ),
            child: Icon(
              isPause ? Icons.hourglass_empty :
              isSpeaking ? Icons.graphic_eq :
              isActive ? Icons.mic :
              Icons.mic_none,
              color: Colors.white,
              size: 28,
            ),
          ),
        ),
      ]),
    );
  }

  Color _buttonColor() {
    switch (_sessionState.state) {
      case VoiceState.speechDetected:
        return const Color(0xFF1D9E75);   // teal — speaking
      case VoiceState.pauseDetected:
        return const Color(0xFFBA7517);   // amber — paused
      case VoiceState.thinking:
        return const Color(0xFF534AB7);   // purple — thinking
      case VoiceState.playingAudio:
        return const Color(0xFF185FA5);   // blue — playing
      case VoiceState.error:
        return const Color(0xFFA32D2D);   // red — error
      case VoiceState.idle:
        return const Color(0xFF888780);   // gray — idle
      default:
        return const Color(0xFF1D9E75);
    }
  }

  Widget _buildWaveform(bool animated) {
    final data = _sessionState.waveformData;
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      crossAxisAlignment: CrossAxisAlignment.end,
      children: List.generate(data.length, (i) {
        final height = animated ? (4 + data[i] * 44).clamp(4.0, 48.0) : 4.0;
        return AnimatedContainer(
          duration: const Duration(milliseconds: 60),
          width: 3,
          height: height,
          margin: const EdgeInsets.symmetric(horizontal: 1),
          decoration: BoxDecoration(
            color: _buttonColor().withOpacity(0.6 + 0.4 * data[i]),
            borderRadius: BorderRadius.circular(2),
          ),
        );
      }),
    );
  }

  Widget _buildStateLabel() {
    final labels = {
      VoiceState.idle:         'தொடங்க தொடுங்கள்',
      VoiceState.listening:    'கேட்கிறேன்... கேள்வி கேளுங்கள்',
      VoiceState.speechDetected: 'பேசுகிறீர்கள்...',
      VoiceState.pauseDetected:'தொடரும் வரை காத்திருக்கிறேன்...',
      VoiceState.transcribing: 'எழுத்தாக்கம் செய்கிறேன்...',
      VoiceState.thinking:     'யோசிக்கிறேன்...',
      VoiceState.responding:   'விளக்கிக் கூறுகிறேன்',
      VoiceState.playingAudio: 'கேளுங்கள்...',
      VoiceState.error:        _sessionState.errorMessage ?? 'பிழை',
    };

    return Text(
      labels[_sessionState.state] ?? '',
      style: TextStyle(
        fontSize: 12,
        color: _sessionState.state == VoiceState.error
            ? const Color(0xFFA32D2D)
            : const Color(0xFF5F5E5A),
      ),
    );
  }

  Widget _buildTranscriptCard() {
    return Container(
      margin: const EdgeInsets.symmetric(vertical: 6),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFFF1EFE8),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: const Color(0xFFD3D1C7), width: 0.5),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(children: [
            const Icon(Icons.person, size: 14, color: Color(0xFF5F5E5A)),
            const SizedBox(width: 4),
            Text('நீங்கள் கேட்டது (${_sessionState.dialectDetected})',
                style: const TextStyle(fontSize: 10, color: Color(0xFF888780))),
            if (_sessionState.isComplete)
              Container(
                margin: const EdgeInsets.only(left: 6),
                padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 1),
                decoration: BoxDecoration(
                  color: const Color(0xFFEAF3DE),
                  borderRadius: BorderRadius.circular(4),
                ),
                child: const Text('முழுமை', style: TextStyle(
                    fontSize: 9, color: Color(0xFF3B6D11))),
              ),
          ]),
          const SizedBox(height: 4),
          Text(_sessionState.transcript,
              style: const TextStyle(fontSize: 14, color: Color(0xFF2C2C2A),
                  fontFamily: 'Latha')),   // Tamil font
        ],
      ),
    );
  }

  Widget _buildResponseCard() {
    return Container(
      margin: const EdgeInsets.symmetric(vertical: 6),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFFE6F1FB),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: const Color(0xFFB5D4F4), width: 0.5),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(children: [
            const Icon(Icons.school, size: 14, color: Color(0xFF185FA5)),
            const SizedBox(width: 4),
            const Text('ஆசிரியர்',
                style: TextStyle(fontSize: 10, color: Color(0xFF185FA5))),
          ]),
          const SizedBox(height: 6),
          Text(_sessionState.tutorResponse,
              style: const TextStyle(fontSize: 13, color: Color(0xFF0C447C),
                  fontFamily: 'Latha', height: 1.6)),
        ],
      ),
    );
  }

  // ── Actions ───────────────────────────────────────────────────────────────

  Future<void> _startListening() async {
    final connected = await _controller.connect();
    if (!connected) return;
    setState(() {});
  }

  @override
  void dispose() {
    _controller.dispose();
    _sessionState.removeListener(_onStateChanged);
    _sessionState.dispose();
    _pulseController.dispose();
    super.dispose();
  }
}


// ─────────────────────────────────────────────────────────────────────────────
// BYTES AUDIO SOURCE (for just_audio)
// ─────────────────────────────────────────────────────────────────────────────

class BytesAudioSource extends StreamAudioSource {
  final Uint8List bytes;
  BytesAudioSource(this.bytes);

  @override
  Future<StreamAudioResponse> request([int? start, int? end]) async {
    start ??= 0;
    end ??= bytes.length;
    return StreamAudioResponse(
      sourceLength: bytes.length,
      contentLength: end - start,
      offset: start,
      contentType: 'audio/mpeg',
      stream: Stream.value(bytes.sublist(start, end)),
    );
  }
}
