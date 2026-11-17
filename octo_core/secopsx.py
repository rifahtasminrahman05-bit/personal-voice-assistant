# -*- coding: utf-8 -*-
"""Advanced Cybersecurity Intelligence (features 37-44).

ThreatIntelCorrelator, SecurityPostureAnalyzer, ConfigRiskDetector,
VulnerabilityPrioritizer, ZeroTrustAdvisor, IncidentResponsePlanner,
CloudSecurityAnalyzer, SupplyChainAnalyzer. Defensive tooling for
authorized systems only.
"""
SEV_ORDER = ("low", "medium", "high", "critical")


class ThreatIntelCorrelator:
    """37. Threat Intelligence Correlation: merge IOCs across sources."""

    def correlate(self, reports):
        by_ioc = {}
        for r in reports:
            for ioc in r.get("iocs", []):
                entry = by_ioc.setdefault(ioc, {"ioc": ioc, "sources": [],
                                                "severity": "low"})
                entry["sources"].append(r["source"])
                sev = r.get("severity", "low")
                if SEV_ORDER.index(sev) > SEV_ORDER.index(entry["severity"]):
                    entry["severity"] = sev
        correlated = [e for e in by_ioc.values() if len(e["sources"]) > 1]
        return {"total_iocs": len(by_ioc),
                "correlated": sorted(correlated,
                                     key=lambda e: (-len(e["sources"]),
                                                    e["ioc"]))}


class SecurityPostureAnalyzer:
    """38. Security Posture Analyzer: controls coverage score."""

    CONTROLS = ("mfa", "encryption_at_rest", "encryption_in_transit",
                "patching", "backups", "logging", "incident_response_plan",
                "least_privilege")

    def assess(self, present_controls):
        present = set(present_controls)
        missing = [c for c in self.CONTROLS if c not in present]
        score = round(len(present & set(self.CONTROLS)) / len(self.CONTROLS), 2)
        grade = ("strong" if score >= 0.85
                 else "moderate" if score >= 0.6 else "weak")
        return {"score": score, "grade": grade, "missing": missing}


class ConfigRiskDetector:
    """39. Configuration Risk Detector: risky config keys/values."""

    RULES = (
        ("debug", "true", "debug enabled in configuration", "medium"),
        ("password", None, "plaintext credential in config", "high"),
        ("secret", None, "plaintext secret in config", "high"),
        ("ssl_verify", "false", "TLS verification disabled", "high"),
        ("public", "true", "public exposure enabled", "high"),
        ("cors", "*", "permissive CORS wildcard", "medium"),
        ("root_login", "true", "direct root access allowed", "high"),
    )

    def scan(self, config):
        findings = []
        for key, value in config.items():
            sval = str(value).lower()
            for rule_key, rule_val, label, sev in self.RULES:
                if rule_key in str(key).lower() and \
                        (rule_val is None or sval == rule_val or sval == "*"):
                    findings.append({"key": key, "value": value,
                                     "finding": label, "severity": sev})
        return {"findings": findings,
                "risk_score": sum(3 if f["severity"] == "high" else 2
                                  for f in findings)}


class VulnerabilityPrioritizer:
    """40. Vulnerability Prioritization: rank by impact x exploitability."""

    SEV = {"low": 1, "medium": 2, "high": 3, "critical": 4}

    def prioritize(self, vulns):
        rows = []
        for v in vulns:
            score = self.SEV.get(v.get("severity", "medium"), 2) * 2.5
            score += 3 if v.get("exploit_available") else 0
            score += 2 if v.get("internet_facing") else 0
            score += v.get("asset_value", 1)
            rows.append({**v, "priority": round(score, 1)})
        rows.sort(key=lambda r: -r["priority"])
        return {"ranked": rows, "top": rows[0]["id"] if rows else None}


class ZeroTrustAdvisor:
    """41. Zero-Trust Policy Intelligence: evaluate access rules."""

    def evaluate(self, rules):
        violations = []
        for r in rules:
            issues = []
            if not r.get("mfa"):
                issues.append("no MFA required")
            if not r.get("device_check"):
                issues.append("no device posture check")
            if r.get("scope") == "*":
                issues.append("over-broad scope")
            if issues:
                violations.append({"rule": r, "issues": issues})
        return {"rules": len(rules),
                "compliant": len(rules) - len(violations),
                "violations": violations,
                "recommendations": ["require MFA everywhere",
                                    "scope least-privilege",
                                    "verify device health"] if violations else []}


class IncidentResponsePlanner:
    """42. Security Incident Response Planner: type-specific playbooks."""

    PLAYBOOKS = {
        "ransomware": ["isolate affected hosts", "identify patient zero",
                       "assess backup integrity", "notify stakeholders",
                       "restore from clean backups"],
        "data_breach": ["contain access", "scope exposed data",
                        "rotate credentials",
                        "notify affected parties per policy",
                        "preserve forensic evidence"],
        "ddos": ["enable rate limiting", "engage scrubbing service",
                 "scale ingress", "monitor recovery"],
        "phishing": ["block sender domain",
                     "force password reset for clickers",
                     "scan for payload execution", "awareness follow-up"],
    }

    def plan(self, incident_type, severity="medium"):
        steps = self.PLAYBOOKS.get(incident_type.lower(),
                                   ["triage", "contain", "eradicate",
                                    "recover", "post-mortem"])
        if severity in ("high", "critical"):
            steps = ["declare incident", "page on-call lead"] + list(steps)
        return {"incident": incident_type, "severity": severity,
                "steps": steps}


class CloudSecurityAnalyzer:
    """43. Cloud Security Intelligence: misconfiguration scan (defensive)."""

    def scan(self, resources):
        findings = []
        for r in resources:
            if r.get("public"):
                findings.append({"resource": r["name"],
                                 "finding": "publicly exposed",
                                 "severity": "high"})
            if r.get("encrypted") is False:
                findings.append({"resource": r["name"],
                                 "finding": "encryption disabled",
                                 "severity": "high"})
            if r.get("logging") is False:
                findings.append({"resource": r["name"],
                                 "finding": "access logging disabled",
                                 "severity": "medium"})
        critical = any(f["severity"] == "high" for f in findings)
        return {"resources": len(resources), "findings": findings,
                "verdict": "needs-attention" if critical else "acceptable"}


class SupplyChainAnalyzer:
    """44. Supply Chain Security Analyzer: dependency provenance risks."""

    RISKY_LICENSES = ("unknown", "unlicensed", "gpl-3.0")

    def analyze(self, packages):
        flagged = []
        for p in packages:
            risks = []
            if not p.get("pinned"):
                risks.append("unpinned version")
            if p.get("signed") is False:
                risks.append("unsigned artifact")
            if p.get("maintainers", 1) <= 1:
                risks.append("single maintainer (bus factor)")
            license_id = str(p.get("license", "unknown")).lower()
            if license_id in self.RISKY_LICENSES:
                risks.append(f"license concern: {license_id}")
            if risks:
                flagged.append({"package": p["name"], "risks": risks})
        return {"packages": len(packages), "flagged": flagged,
                "clean": len(packages) - len(flagged)}
