def truncate_long_text(text, max_length=200):
    return (text[:max_length] + '..') if len(text) > max_length else text

def truncate_messages(messages):
    return [{k: v if k == "role" else truncate_long_text(v)} for i in messages for k, v in i.items()]