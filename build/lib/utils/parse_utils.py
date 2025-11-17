
def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None


def last_boxed_only(sample):
    """
    Given a (q,a) sample, filter the answers so that they only contain 
    the last \boxed{...} or \fbox{...} element
    """
    q, a = sample
    a = last_boxed_only_string(a)
    if a is None:
        return None
    return (q, a)

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
        

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval

def extract_sentence(input_str, substring):
    """
    Extracts the entire sentence containing the given substring from the input string.
    Returns the sentence if found, otherwise None.
    """
    import re
    # Split the input string into sentences based on punctuation followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', input_str)
    for sentence in sentences:
        if substring in sentence:
            return sentence.strip()
    return None

def parse_answer(input_str, answer_string):
    # Try finding the boxed answer first 
    boxed_answer = remove_boxed(last_boxed_only_string(input_str))

    # If the LLM didn't provide a boxed answer, try to see if it mentioned the correct answer
    if(boxed_answer == "{}" or boxed_answer is None):
        context_answer = extract_sentence(input_str, answer_string)
        return context_answer

    return boxed_answer
