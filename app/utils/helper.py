def people_changed(current, last):
    if len(current) != len(last):
        return True
    for c, l in zip(current, last):
        if c["id"] != l["id"] or (c["name"] != l["name"] and c["name"] != "Unknown"):
            return True
    return False