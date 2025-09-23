"""
Model-specific name extraction methods for different AI models.
Each model may have different output formats and require different extraction strategies.
"""

import re

# List of all possible friend names (in alphabetical order for consistent binary representation)
ALL_FRIENDS = [
    'Alice', 'Bob', 'Carol', 'Dave', 'Erin', 'Frank', 'Grace', 'Heidi', 
    'Ivan', 'Judy', 'Mallory', 'Niaj', 'Olivia', 'Peggy', 'Quentin', 
    'Rupert', 'Sybil', 'Trent', 'Uma', 'Victor'
]

def extract_names_gemini_2_5_pro(text, num_friends=None):
    """
    Extract friend names from Gemini 2.5 Pro model generation text.
    
    Gemini 2.5 Pro tends to use structured reasoning with clear conclusions,
    often with bold text formatting and specific conclusion phrases.
    
    Args:
        text (str): The model generation text
        num_friends (int): Number of friends in the dataset (for validation)
        
    Returns:
        list: List of friend names that raise their hands
    """
    if num_friends is None:
        num_friends = len(ALL_FRIENDS)
    
    # Get the relevant friends for this dataset
    relevant_friends = ALL_FRIENDS[:num_friends]
    
    # First check for "no one" or "nobody" patterns
    no_one_patterns = [
        r'no\s+one\s+(?:raises?|will\s+raise|raised)\s+(?:a\s+|their\s+)?hands?',
        r'nobody\s+(?:raises?|will\s+raise|raised)\s+(?:their\s+)?hands?',
        r'no\s+friends?\s+(?:raise|will\s+raise|raised)\s+(?:their\s+)?hands?',
        r'everyone\s+(?:keeps|kept)\s+(?:their\s+)?hands?\s+(?:down|on\s+the\s+table)',
        r'all\s+friends?\s+(?:keep|kept)\s+(?:their\s+)?hands?\s+(?:down|on\s+the\s+table)',
        r'no\s+hands?\s+(?:are\s+)?(?:raised|up)',
        # More specific patterns for "all hands down" meaning no one raises hands
        r'(?:all\s+)?hands?\s+(?:are\s+)?(?:all\s+)?(?:down|on\s+the\s+table)\s+(?:in\s+Round|for\s+Round|this\s+round)',
        # Additional patterns for "no one" cases
        r'therefore,?\s+in\s+round\s+\d+:\s*\*\*no\s+one\s+raises?\s+(?:a\s+|their\s+)?hands?\*\*',
        r'therefore,?\s+in\s+round\s+\d+:\s*no\s+one\s+raises?\s+(?:a\s+|their\s+)?hands?',
        r'\*\*no\s+one\s+raises?\s+(?:a\s+|their\s+)?hands?\*\*',
        r'no\s+one\s+raises?\s+(?:a\s+|their\s+)?hands?\.?$',
        # Pattern for "Therefore, in Round 6:\n**No one raises a hand.**"
        r'therefore,?\s+in\s+round\s+\d+:\s*\n\*\*no\s+one\s+raises?\s+(?:a\s+|their\s+)?hands?\*\*',
    ]
    
    text_lower = text.lower()
    for pattern in no_one_patterns:
        if re.search(pattern, text_lower):
            return []  # No one raises their hand
    
    # Look for the final answer section - Gemini 2.5 Pro often uses structured conclusions
    conclusion_phrases = [
        r'Therefore,?\s+in\s+Round\s+\d+',
        r'So,?\s+in\s+Round\s+\d+',
        r'In\s+Round\s+\d+',
        r'Therefore,?\s+the\s+following',
        r'So,?\s+the\s+following',
        r'Following\s+this\s+rule',
        r'By\s+applying\s+this\s+rule',
        r'Based\s+on\s+the\s+pattern',
        r'Following\s+the\s+requested\s+format',
        r'Here\s+is\s+what\s+will\s+happen',
        r'Here\'s\s+what\s+will\s+happen',
        r'The\s+answer\s+is:',
        r'Following\s+the\s+requested\s+format,\s+the\s+answer\s+is:',
        # Gemini 2.5 Pro specific patterns for multi-round reasoning
        r'And\s+finally,?\s+for\s+the\s+round\s+in\s+question:',
        r'Finally,?\s+for\s+the\s+round\s+in\s+question:',
        r'And\s+finally,?\s+in\s+Round\s+\d+:',
        r'Finally,?\s+in\s+Round\s+\d+:',
    ]
    
    # Find the last occurrence of any conclusion phrase
    # Prioritize "finally" and "for the round in question" patterns (Gemini 2.5 Pro specific)
    priority_phrases = [
        r'And\s+finally,?\s+for\s+the\s+round\s+in\s+question:',
        r'Finally,?\s+for\s+the\s+round\s+in\s+question:',
        r'And\s+finally,?\s+in\s+Round\s+\d+:',
        r'Finally,?\s+in\s+Round\s+\d+:',
    ]
    
    last_conclusion_pos = -1
    # First check for priority phrases - if found, use the latest one
    priority_matches = []
    for phrase in priority_phrases:
        matches = list(re.finditer(phrase, text, re.IGNORECASE))
        priority_matches.extend(matches)
    
    if priority_matches:
        # Use the latest priority match
        last_conclusion_pos = max(match.start() for match in priority_matches)
    else:
        # If no priority phrases found, check regular conclusion phrases
        for phrase in conclusion_phrases:
            matches = list(re.finditer(phrase, text, re.IGNORECASE))
            if matches:
                last_conclusion_pos = max(last_conclusion_pos, matches[-1].start())
    
    # If we found a conclusion, focus on text after it
    if last_conclusion_pos >= 0:
        conclusion_text = text[last_conclusion_pos:]
    else:
        # Look at the last 500 characters
        conclusion_text = text[-500:]
    
    # Gemini 2.5 Pro specific patterns - often uses bold text and structured formatting
    patterns = [
        # Simple bold text pattern - extract everything in bold
        r'\*\*([^*]+?)\*\*',
        # Bold text patterns - extract everything in bold, but be more specific
        r'\*\*([^*]+?)\s+(?:raise|will\s+raise|raised)\s+(?:their\s+)?hands?\*\*',
        # Lists of names followed by "raise their hands" - more specific
        r'([A-Za-z,\s]+?)\s+(?:raise|will\s+raise|raised)\s+(?:their\s+)?hands?\.?$',
        # Names in parentheses or after colons
        r':\s*([A-Za-z,\s]+)',
        # Direct name lists with "and" - be more specific
        r'([A-Za-z,\s]+?)\s+and\s+([A-Za-z,\s]+?)\s+(?:raise|will\s+raise|raised)',
        # Simple comma-separated lists ending with "raise their hands"
        r'([A-Za-z,\s]+?)\s+(?:raise|will\s+raise|raised)\s+(?:their\s+)?hands?\.?$',
        # Pattern for "Alice, Bob, Carol, Erin, Frank, and Grace raise their hands"
        r'([A-Za-z,\s]+?)\s+and\s+([A-Za-z,\s]+?)\s+(?:raise|will\s+raise|raised)\s+(?:their\s+)?hands?\.?$',
        # Pattern for "In Round X, **names** will raise their hands" - handle the "In Round X," prefix
        r'In\s+Round\s+\d+,?\s*\*\*([^*]+?)\s+(?:raise|will\s+raise|raised)\s+(?:their\s+)?hands?\*\*',
        # Pattern for "In Round X, names will raise their hands" - without bold
        r'In\s+Round\s+\d+,?\s*([A-Za-z,\s]+?)\s+(?:raise|will\s+raise|raised)\s+(?:their\s+)?hands?\.?$',
        # Pattern for "In Round X, **names** will raise their hands" - more flexible
        r'In\s+Round\s+\d+,?\s*\*\*([^*]+?)\s+will\s+raise\s+(?:their\s+)?hands?\*\*',
        # Pattern for "In Round X, names will raise their hands" - more flexible
        r'In\s+Round\s+\d+,?\s*([A-Za-z,\s]+?)\s+will\s+raise\s+(?:their\s+)?hands?\.?$',
    ]
    
    found_names = []
    
    # Try each pattern on the conclusion text
    for pattern in patterns:
        matches = re.findall(pattern, conclusion_text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            if isinstance(match, tuple):
                # Handle tuple matches by joining
                match = ' '.join(match)
            
            # Clean up the match and split by comma
            # First remove "raise their hands" and similar phrases
            names_str = re.sub(r'\s+(?:raise|will\s+raise|raised)\s+(?:their\s+)?hands?.*$', '', match.strip())
            # Then remove non-word characters except commas, spaces, and "and"
            names_str = re.sub(r'[^\w,\s]', '', names_str)
            # Clean up extra whitespace and newlines
            names_str = re.sub(r'\s+', ' ', names_str).strip()
            
            # Split by comma and clean up each name
            names = []
            for name in names_str.split(','):
                name = name.strip()
                if name:
                    # Handle "and Victor" case by splitting on "and"
                    if ' and ' in name:
                        parts = name.split(' and ')
                        for part in parts:
                            part = part.strip()
                            if part and not part.startswith('and '):
                                names.append(part)
                    else:
                        # Also check for names that start with "and "
                        if name.startswith('and '):
                            name = name[4:].strip()
                        if name:
                            names.append(name)
            
            # Also handle cases where names are separated by "and" without commas
            if not names and ' and ' in names_str:
                # Split by "and" and clean up
                and_parts = names_str.split(' and ')
                for part in and_parts:
                    part = part.strip()
                    if part:
                        names.append(part)
            
            # Check if these are valid friend names
            valid_names = []
            for name in names:
                if name in relevant_friends:
                    valid_names.append(name)
            
            # If we found a reasonable number of valid names, use this result
            if len(valid_names) >= 1:
                found_names = valid_names
                break
        
        if found_names:
            break
    
    # If still no names found, try the fallback approach on the full text
    if not found_names:
        fallback_patterns = [
            # Pattern for "Frank, Grace, and Alice raise their hands"
            r'([A-Za-z,\s]+?)\s+and\s+([A-Za-z,\s]+?)\s+(?:raise|will\s+raise|raised)\s+(?:their\s+)?hands?',
            # Pattern for "Frank, Grace, and Alice raise their hands." (with period)
            r'([A-Za-z,\s]+?)\s+and\s+([A-Za-z,\s]+?)\s+(?:raise|will\s+raise|raised)\s+(?:their\s+)?hands?\.?$',
            # Pattern for comma-separated lists followed by "raise their hands"
            r'([A-Za-z,\s]+?)\s+(?:raise|will\s+raise|raised)\s+(?:their\s+)?hands?\.?$',
            # Original patterns
            r'(?:raise their hands?|will raise their hands?|raise their hand|will raise their hand)[:.]?\s*([A-Za-z,\s]+)',
            r'(?:hands? up|hands up)[:.]?\s*([A-Za-z,\s]+)',
        ]
        
        for pattern in fallback_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if isinstance(match, tuple):
                    # Handle tuple matches by joining
                    match = ' '.join(match)
                
                # Clean up the match
                names_str = re.sub(r'\s+(?:raise|will\s+raise|raised)\s+(?:their\s+)?hands?.*$', '', match.strip())
                names_str = re.sub(r'[^\w,\s]', '', names_str)
                # Clean up extra whitespace and newlines
                names_str = re.sub(r'\s+', ' ', names_str).strip()
                
                # Split by comma and clean up each name
                names = []
                for name in names_str.split(','):
                    name = name.strip()
                    if name:
                        # Handle "and Victor" case by splitting on "and"
                        if ' and ' in name:
                            parts = name.split(' and ')
                            for part in parts:
                                part = part.strip()
                                if part and not part.startswith('and '):
                                    names.append(part)
                        else:
                            # Also check for names that start with "and "
                            if name.startswith('and '):
                                name = name[4:].strip()
                            if name:
                                names.append(name)
                
                # Also handle cases where names are separated by "and" without commas
                if not names and ' and ' in names_str:
                    # Split by "and" and clean up
                    and_parts = names_str.split(' and ')
                    for part in and_parts:
                        part = part.strip()
                        if part:
                            names.append(part)
                
                valid_names = []
                for name in names:
                    if name in relevant_friends:
                        valid_names.append(name)
                
                if len(valid_names) >= 1:
                    found_names = valid_names
                    break
            
            if found_names:
                break
    
    # Final fallback: look for any friend names mentioned
    if not found_names:
        for friend in relevant_friends:
            if friend.lower() in text_lower:
                found_names.append(friend)
    
    return found_names


def parse_target_string(target_str):
    """Parse the target string to get list of friend names"""
    # Handle empty or None target strings
    if not target_str or target_str.strip() == '':
        return []
    
    # Split by comma and clean up whitespace
    names = [name.strip() for name in target_str.split(',') if name.strip()]
    return names


def extract_names_generic(text, num_friends=None):
    """
    Generic name extraction method for models with unknown output formats.
    Uses a simpler, more general approach.
    
    Args:
        text (str): The model generation text
        num_friends (int): Number of friends in the dataset (for validation)
        
    Returns:
        list: List of friend names that raise their hands
    """
    if num_friends is None:
        num_friends = len(ALL_FRIENDS)
    
    # Get the relevant friends for this dataset
    relevant_friends = ALL_FRIENDS[:num_friends]
    
    # Simple patterns for generic extraction
    patterns = [
        r'([A-Za-z,\s]+?)\s+(?:raise|will\s+raise|raised)\s+(?:their\s+)?hands?\.?$',
        r'(?:raise their hands?|will raise their hands?|raise their hand|will raise their hand)[:.]?\s*([A-Za-z,\s]+)',
        r'(?:hands? up|hands up)[:.]?\s*([A-Za-z,\s]+)',
    ]
    
    found_names = []
    text_lower = text.lower()
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            if isinstance(match, tuple):
                match = ' '.join(match)
            
            # Clean up the match
            names_str = re.sub(r'\s+(?:raise|will\s+raise|raised)\s+(?:their\s+)?hands?.*$', '', match.strip())
            names_str = re.sub(r'[^\w,\s]', '', names_str)
            names_str = re.sub(r'\s+', ' ', names_str).strip()
            
            # Split by comma and clean up each name
            names = []
            for name in names_str.split(','):
                name = name.strip()
                if name and name in relevant_friends:
                    names.append(name)
            
            if names:
                found_names = names
                break
        
        if found_names:
            break
    
    return found_names


def get_extractor_for_model(model_name):
    """
    Get the appropriate name extractor function for a given model.
    
    Args:
        model_name (str): Name of the model (e.g., 'gemini-2.5-pro')
        
    Returns:
        function: The appropriate extraction function
    """
    if 'gemini-2.5-pro' in model_name.lower():
        return extract_names_gemini_2_5_pro
    else:
        # Default to generic extractor for unknown models
        return extract_names_generic


# For backward compatibility, provide a default extractor
def extract_names_from_text(text, num_friends=None, model_name='gemini-2.5-pro'):
    """
    Main extraction function that routes to model-specific extractors.
    
    Args:
        text (str): The model generation text
        num_friends (int): Number of friends in the dataset (for validation)
        model_name (str): Name of the model to use appropriate extractor
        
    Returns:
        list: List of friend names that raise their hands
    """
    extractor = get_extractor_for_model(model_name)
    return extractor(text, num_friends)
