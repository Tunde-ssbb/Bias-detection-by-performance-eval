
def get_pairs(records: list[dict], mode: str = "occ_match") -> list[tuple[str, str]]:
    cvs = [r for r in records if r.get("custom_id", "").startswith("cv::")]
    jds = [r for r in records if r.get("custom_id", "").startswith("jd::")]

    if mode == "all":
        return [(cv["custom_id"], jd["custom_id"]) for cv in cvs for jd in jds]

    if mode == "occ_match":
        jds_by_occ: dict[str, list] = {}
        for jd in jds:
            jds_by_occ.setdefault(jd["occ_code"], []).append(jd["custom_id"])
        return [
            (cv["custom_id"], jd_id)
            for cv in cvs
            for jd_id in jds_by_occ.get(cv["occ_code"], [])
        ]

    raise ValueError(f"Unknown pairing mode: {mode!r}. Choose 'occ_match' or 'all'.")
