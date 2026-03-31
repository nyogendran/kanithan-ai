"""
Prerequisite dependency graph and topic-to-skill mapping.
Single source of truth for the pedagogical progression.
"""

from __future__ import annotations

TOPIC_TO_SKILL: dict[str, str] = {
    "divisibility_2": "divisibility_rules",
    "divisibility_3": "divisibility_rules",
    "divisibility_4": "divisibility_rules",
    "divisibility_5": "divisibility_rules",
    "divisibility_6": "divisibility_rules",
    "divisibility_9": "divisibility_rules",
    "divisibility_10": "divisibility_rules",
    "divisibility_rules": "divisibility_rules",
    "divisibility_rules_all": "divisibility_rules",
    "digit_sum": "digit_sum",
    "factor_definition": "factor_listing",
    "factor_listing": "factor_listing",
    "factor_listing_pair_method": "factor_listing",
    "factor_pairs": "factor_listing",
    "prime_number_definition": "prime_factorization",
    "prime_factorization": "prime_factorization",
    "prime_factorization_tree": "prime_factorization",
    "prime_factorization_division": "prime_factorization",
    "factors_via_prime": "prime_factorization",
    "hcf": "hcf",
    "hcf_definition": "hcf",
    "hcf_method_1_list": "hcf",
    "hcf_method_2_prime": "hcf",
    "hcf_method_3_division": "hcf",
    "hcf_word_problem": "hcf",
    "hcf_lcm_relationship": "hcf",
    "lcm": "lcm",
    "lcm_definition": "lcm",
    "lcm_prime_method": "lcm",
    "lcm_division_method": "lcm",
    "lcm_word_problem": "lcm",
    "word_problems": "word_problems",
    "word_problem": "word_problems",
}

PREREQUISITE_GRAPH: dict[str, list[str]] = {
    "digit_sum": ["divisibility_rules"],
    "factor_listing": ["divisibility_rules"],
    "factor_listing_pair_method": ["factor_definition"],
    "prime_number_definition": ["factor_definition"],
    "prime_factorization_tree": ["factor_listing", "divisibility_rules"],
    "prime_factorization_division": ["prime_number_definition"],
    "hcf_definition": ["factor_listing", "prime_factorization_tree"],
    "hcf_method_1_list": ["hcf_definition"],
    "hcf_method_2_prime": ["hcf_definition", "prime_factorization_tree"],
    "hcf_method_3_division": ["hcf_definition", "prime_factorization_division"],
    "lcm_definition": ["factor_listing", "prime_factorization_tree"],
    "lcm_prime_method": ["lcm_definition", "prime_factorization_tree"],
    "lcm_division_method": ["lcm_definition", "prime_factorization_division"],
    "word_problems": ["hcf_definition", "lcm_definition"],
}

SECTION_TOPIC_MAP: dict[str, str] = {
    "divisibility_rules": "4.1",
    "digit_sum": "4.1",
    "factor_listing": "4.2",
    "factor_definition": "4.1",
    "prime_factorization": "4.3",
    "prime_number_definition": "4.3",
    "factors_via_prime": "4.4",
    "hcf": "4.5",
    "hcf_definition": "4.5",
    "lcm": "4.6",
    "lcm_definition": "4.6",
    "word_problems": "4.6",
    "word_problem": "4.6",
}


def topic_to_skill(topic: str) -> str:
    """Resolve any topic name to its parent skill key."""
    return TOPIC_TO_SKILL.get(topic, "divisibility_rules")
