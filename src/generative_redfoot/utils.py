def truncate_long_text(text, max_length=200):
    return (text[:max_length] + '..') if len(text) > max_length else text
