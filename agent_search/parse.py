def extract_boxed(resp: str) -> str:
    """Strictly extract the boxed answer from the response."""
    if "oxed{" not in resp:
        return None
    ans = resp.split("oxed{")[-1]
    stack = 1
    a = ""
    for i_pre, c in enumerate(ans):
        if ans[i_pre] == "\\":
            a += c
            continue
        if c == "{":
            stack += 1
            a += c
        elif c == "}":
            stack -= 1
            if stack == 0:
                break
            a += c
        else:
            a += c

    return a


def extract_ans_is(resp: str) -> str:
    """Extract the answer from the response."""
    if "nswer is" not in resp:
        return None
    ans = resp.split("is")[-1].lstrip(":$ ").rstrip(".:$ ")
    return ans


def extract_ans(resp: str) -> str:
    """Extract the answer from the response."""
    ans = extract_boxed(resp)
    if ans is None:
        ans = extract_ans_is(resp)
    return ans
