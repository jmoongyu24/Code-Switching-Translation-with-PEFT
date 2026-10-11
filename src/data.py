"""데이터 로드·저장·검증, TED2020 기반 학습 데이터(단일 언어 + 합성 code-switched) 생성함."""
import bisect
import collections
import itertools
import json
import random
import re
from collections import Counter

TARGETS = ("en", "ko")
INPUT_TYPES = ("single", "mixed")
GROUPS = [(t, l) for t in INPUT_TYPES for l in TARGETS]  # (input_type, target_language)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def check_eval_rows(rows, need_answer):
    """필드·값·id 중복 검사하고 group별 개수 반환함."""
    need = ["id", "input", "target_language", "input_type"] + (["answer"] if need_answer else [])
    for r in rows:
        missing = [k for k in need if k not in r]
        if missing:
            raise ValueError(f"id={r.get('id')}: 필드 없음 {missing}")
        if r["target_language"] not in TARGETS or r["input_type"] not in INPUT_TYPES:
            raise ValueError(f"id={r['id']}: 잘못된 값 {r['target_language']}/{r['input_type']}")
        if not r["input"].strip() or (need_answer and not r["answer"].strip()):
            raise ValueError(f"id={r['id']}: 빈 input/answer 있음")
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("id 중복 있음")
    return Counter((r["input_type"], r["target_language"]) for r in rows)


def take_per_group(rows, k):
    """group별 앞 k개만 남김 (smoke test용). 원래 순서 유지함."""
    seen, out = Counter(), []
    for r in rows:
        g = (r["input_type"], r["target_language"])
        if seen[g] < k:
            out.append(r)
            seen[g] += 1
    return out


# ---------------- 학습 데이터: TED2020 en 기준 join으로 en/es/ko/ja 4개 언어 pool 구성 ----------------
LANGS = ["en", "es", "ko", "ja"]
DIRECTIONS = [("ko", "en"), ("es", "en"), ("ja", "en"), ("en", "ko"), ("es", "ko"), ("ja", "ko")]  # val single과 같음
PUNCT = set(",.!?;:、。！？；：…\"'“”‘’「」『』¿¡")
CLAUSE_P = (",", "、", ";", "，")
HANGUL, KANA, HAN = r"[가-힣]", r"[぀-ヿ]", r"[一-鿿]"
SCRIPT = {"ko": HANGUL, "ja": f"{KANA}|{HAN}", "en": r"[A-Za-z]", "es": r"[A-Za-zÁÉÍÓÚÑáéíóúñü]"}
_SCR = {"en": "lat", "es": "lat", "ko": "ko", "ja": "ja"}


def script_cat(text):
    """입력에 나타난 문자 체계 조합 (ko: 한글, ja: 가나·한자, lat: 라틴). val 분포 비교·할당량에 씀."""
    found = [k for k, p in (("ko", HANGUL), ("ja", f"{KANA}|{HAN}"), ("lat", r"[A-Za-z]")) if re.search(p, text)]
    return "+".join(found)


# 문자 체계 조합별로 가능한 (주 언어, 삽입 언어들) 조합
COMBOS = collections.defaultdict(list)
for _m in ("en", "es", "ko", "ja"):
    for _k in (1, 2):
        for _emb in itertools.combinations([x for x in ("en", "es", "ko", "ja") if x != _m], _k):
            COMBOS["+".join(s for s in ("ko", "ja", "lat") if s in {_SCR[x] for x in (_m, *_emb)})].append((_m, _emb))
BAD = re.compile(r"[()\[\]（）♪♫]|https?://|www\.|--|^\S{1,25}:\s")  # 자막 주석·화자 표시·URL 등
CONJ = {"en": set("but so because when if although though while since until unless whereas".split()),
        "es": set("pero porque cuando si aunque mientras hasta sino pues".split())}
COORD = {"en": {"and", "or"}, "es": {"y", "e", "o", "u"}}  # 쉼표 뒤에 올 때만 절 경계로 봄
STOP = {"en": set("the a an of to and or but in on at for with by from that which who is are was were be as so".split()),
        "es": set("el la los las un una de del a al y o pero en con por para que es son se lo le su sus".split())}
PRON = set("i you he she it we they me him her us them yo tú tu él ella nosotros nosotras ellos ellas usted ustedes".split())
KO_NOUN = ("NNG", "NNP", "NR", "NP", "SL", "SN", "SH", "XSN", "XR")
KO_BAD_IN_NP = ("EF", "EC", "EP", "JKS", "JKC", "JKO", "JKB", "JKV", "JKQ", "JX", "JC", "SF", "SP", "SS")
KO_FORMAL = {"것", "수", "때", "중", "데", "바", "뿐", "등", "거", "줄", "적", "만큼"}
JA_CONTENT = {"名詞", "動詞", "形容詞", "形状詞", "副詞", "連体詞", "接頭辞", "接尾辞", "代名詞", "感動詞"}
JA_FORMAL = {"ため", "こと", "もの", "ところ", "よう", "とき", "時", "ほう", "方", "の", "わけ", "はず"}


def load_pool(raw_dir, max_en):
    """en-ko/en-es/en-ja를 같은 영어 문장으로 join하고 규칙 필터 적용함. 영어가 중복된 쌍은 버림."""
    import pyarrow.parquet as pq
    tabs = {}
    for x in ("ko", "es", "ja"):
        rows = pq.read_table(f"{raw_dir}/en-{x}/train-00000-of-00001.parquet").to_pylist()
        c = Counter(r["english"].strip() for r in rows)
        tabs[x] = {r["english"].strip(): " ".join(r["non_english"].split()) for r in rows if c[r["english"].strip()] == 1}
    out = []
    for en in tabs["ko"].keys() & tabs["es"].keys() & tabs["ja"].keys():
        r = {"en": " ".join(en.split()), "ko": tabs["ko"][en], "es": tabs["es"][en], "ja": tabs["ja"][en]}
        if keep_row(r, max_en):
            out.append(r)
    out.sort(key=lambda r: r["en"])  # set 순서 의존 제거함
    return out


def keep_row(r, max_en):
    en, ko, es, ja = r["en"], r["ko"], r["es"], r["ja"]
    if not 20 <= len(en) <= max_en or any(BAD.search(x) for x in r.values()):
        return False
    if not re.search(HANGUL, ko) or re.search(KANA, ko) or not re.search(KANA, ja) or re.search(HANGUL, ja):
        return False
    if re.search(f"{HANGUL}|{KANA}|{HAN}", en + es) or es == en:
        return False
    for x in r.values():  # 한 문장만 허용함, 따옴표 짝 맞아야 함
        if re.search(r"[.!?。！？](?=.)", x.rstrip(".!?。！？\"'”’」 ")) or x[0] in ",.;:、。":
            return False
        if x.count('"') % 2 or x.count("“") != x.count("”") or x.count("「") != x.count("」"):
            return False
    return 0.2 <= len(ko) / len(en) <= 1.2 and 0.15 <= len(ja) / len(en) <= 1.2 and 0.6 <= len(es) / len(en) <= 2.0


class Analyzer:
    """형태소 분석 (ko: Kiwi, ja: MeCab/unidic-lite). en/es는 공백 단어 단위."""

    def __init__(self):
        import fugashi
        from kiwipiepy import Kiwi
        self.kiwi, self.tagger = Kiwi(), fugashi.Tagger()

    def morphs(self, text, lang):
        """(start, end, tag, surface, cform, pos2, pos3) 리스트 반환함."""
        if lang == "ko":
            return [(t.start, t.start + t.len, t.tag, t.form, "", "", "") for t in self.kiwi.tokenize(text)]
        if lang == "ja":
            out, pos = [], 0
            for w in self.tagger(text):
                s = pos + len(w.white_space)
                e = s + len(w.surface)
                f = w.feature
                out.append((s, e, f.pos1, w.surface, f.cForm or "", f.pos2 or "", f.pos3 or ""))
                pos = e
            return out
        return [(m.start(), m.end(), "W", m.group(), "", "", "") for m in re.finditer(r"\S+", text)]


def complete(text, lang, ms):
    """완결된 문장인지 확인함 (중간에 잘린 자막 조각 제거용)."""
    if lang in ("en", "es"):
        first = text.lstrip("\"“'‘¿¡")[:1]
        return bool(re.search(r"[.!?][\"”’']?$", text)) and (first.isupper() or first.isdigit())
    if lang == "ko":
        body = [m for m in ms if not (m[2].startswith("S") and m[2] not in ("SL", "SN", "SH"))]
        return bool(body) and body[-1][2] == "EF"  # 종결어미로 끝나야 함
    body = [m for m in ms if m[2] not in ("補助記号", "空白", "記号")]
    if not body or (body[-1][2] == "助詞" and body[-1][5] != "終助詞"):
        return False
    return not body[-1][4].startswith("連用形")


class Info:
    """문장 하나의 교체 가능 경계와 품사 정보.
    S: 구간 시작 가능 위치, E: 끝 가능 위치, C: 내용어로 끝나는 위치, P: 뒤에 조사가 붙는 위치,
    CS/CE: 절 시작·끝 위치 (쉼표, 접속사, 연결어미, 접속조사, 일본어 자막 공백 기준)."""

    def __init__(self, text, lang, an):
        self.text, self.lang, self.ms = text, lang, an.morphs(text, lang)
        t = text
        self.S, self.E, self.C, self.P = set(), set(), set(), set()
        self.end = len(t.rstrip("".join(PUNCT) + " "))
        if lang in ("en", "es", "ko"):
            for m in re.finditer(r"\S+", t):
                a, b = m.start(), m.end()
                while a < b and t[a] in PUNCT:
                    a += 1
                while b > a and t[b - 1] in PUNCT:
                    b -= 1
                if a >= b:
                    continue
                self.S.add(a)
                self.E.add(b)
                if lang != "ko":
                    self.C.add(b)
                    continue
                ws = [x for x in self.ms if a <= x[0] < b]
                k = len(ws)
                while k > 0 and ws[k - 1][2].startswith(("JK", "JX", "JC")):
                    k -= 1
                if 0 < k < len(ws) and a < ws[k][0] < b:  # 조사 앞에서 자를 수 있음
                    self.E.add(ws[k][0]); self.C.add(ws[k][0]); self.P.add(ws[k][0])
                elif k == len(ws) and ws and ws[-1][2] in KO_NOUN:
                    self.C.add(b)
        else:
            ms = self.ms
            for i, (s, e, p, sur, cf, p2, p3) in enumerate(ms):
                prev = ms[i - 1][2] if i else None
                nn = ms[i + 1][2] if i + 1 < len(ms) else None
                if p in JA_CONTENT and (prev is None or prev not in JA_CONTENT):
                    self.S.add(s)
                if p in JA_CONTENT and (nn is None or nn not in JA_CONTENT):
                    noun = p in ("名詞", "代名詞", "接尾辞", "形状詞")
                    if nn is None or nn in ("補助記号", "空白"):
                        self.E.add(e)
                        if noun:
                            self.C.add(e)
                    elif nn == "助詞" and noun:
                        self.E.add(e); self.C.add(e); self.P.add(e)
                    elif nn == "助動詞" and p == "名詞":
                        self.E.add(e); self.C.add(e)  # 名詞+です 형태
                if p == "助動詞" and (nn is None or nn not in ("助詞", "助動詞")):
                    self.E.add(e)
                if p == "助詞" and p2 == "接続助詞" and (nn is None or nn not in ("助詞", "助動詞")):
                    self.E.add(e)  # 동사+て/が/から 등 절 끝
        self.CS = {min(self.S)} if self.S else set()
        self.CE = {x for x in self.E if x == self.end} or ({max(self.E)} if self.E else set())
        cuts = {i for i, c in enumerate(t) if c in CLAUSE_P}
        if lang in ("en", "es"):
            for m in re.finditer(r"\S+", t):
                w = re.sub(r"\W", "", m.group().lower())
                coord = w in COORD[lang] and t[:m.start()].rstrip().endswith((",", ";"))
                if (w in CONJ[lang] or coord) and m.start() > 0:
                    cuts.add(m.start() - 1)  # 접속사 앞에서 끝나고 접속사 뒤에서 시작함
                    nxt = [x for x in self.S if x > m.end()]
                    if nxt:
                        self.CS.add(min(nxt))
        elif lang == "ko":
            words = list(re.finditer(r"\S+", t))
            for n, m in enumerate(words):
                ws = [x for x in self.ms if m.start() <= x[0] < m.end() and not x[2].startswith("S")]
                nxt = [x for x in self.ms if n + 1 < len(words) and words[n + 1].start() <= x[0] < words[n + 1].end()]
                if ws and ws[-1][2] == "EC" and m.end() < self.end and not (nxt and nxt[0][2] == "VX"):
                    cuts.add(m.end())  # 보조용언(VX) 앞의 연결어미는 절 경계가 아님 (-고 있다 등)
        else:
            cuts |= {x[1] for x in self.ms if x[2] == "助詞" and x[5] == "接続助詞"}
            cuts |= {i for i, c in enumerate(t) if c == " "}

            def aux_follows(i):  # て+補助動詞(いる·みる·ください 등)나 조사·조동사가 이어지면 절 경계가 아님
                nxt = next((x for x in self.ms if x[0] >= i), None)
                prv = next((x for x in reversed(self.ms) if x[1] <= i), None)
                if nxt is None:
                    return False
                if nxt[2] in ("助詞", "助動詞", "接尾辞"):
                    return True
                return nxt[2] in ("動詞", "形容詞") and nxt[5] == "非自立可能" and prv is not None and prv[5] == "接続助詞"

            cuts = {i for i in cuts if not aux_follows(i)}
        for i in cuts:
            after = [x for x in self.S if x > i]
            before = [x for x in self.E if x <= i]
            if after:
                self.CS.add(min(after))
            if before:
                self.CE.add(max(before))

    def is_np(self, s, e):
        """[s, e)가 명사로 끝나는 명사구인지 확인함. en/es는 품사 정보가 없어 대명사만 거름."""
        if self.lang in ("en", "es"):
            ws = re.findall(r"[^\W\d_]+", self.text[s:e].lower())
            return bool(ws) and not all(w in PRON for w in ws)
        ms = [m for m in self.ms if s <= m[0] < e]
        if not ms:
            return False
        if self.lang == "ko":
            if ms[-1][2] not in KO_NOUN or ms[-1][3] in KO_FORMAL or ms[0][3] in KO_FORMAL:
                return False
            if ms[0][2] in ("JKG", "ETM", "XSN", "XSV", "XSA", "NNB", "EC", "EF", "EP"):
                return False
            if all(m[2] in ("NP", "XSN") for m in ms):
                return False
            return not any(m[2] in KO_BAD_IN_NP for m in ms)
        if ms[-1][2] not in ("名詞", "代名詞", "接尾辞") or ms[-1][3] in JA_FORMAL or ms[-1][6] == "副詞可能":
            return False
        if ms[0][2] in ("動詞", "助動詞", "形容詞", "助詞") or " " in self.text[s:e]:
            return False
        if all(m[2] in ("代名詞", "接尾辞") for m in ms):
            return False
        return not any((m[2] == "助詞" and m[3] != "の") or m[2] == "副詞" or (m[2] == "補助記号" and m[3] != "・")
                       for m in ms)


class Aligner:
    """XLM-R 중간 layer embedding의 양방향 argmax 교집합으로 subword alignment 계산함 (SimAlign argmax 방식)."""

    def __init__(self, name, revision, layer, device="cuda"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name, revision=revision)
        self.model = AutoModel.from_pretrained(name, revision=revision).to(device).eval()
        self.layer, self.device = layer, device
        self.skip = set(self.tok.all_special_ids) | {self.tok.convert_tokens_to_ids("▁")}

    def encode(self, texts, bs=256):
        torch, out = self.torch, []
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                enc = self.tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True,
                               max_length=192, return_offsets_mapping=True)
                offs, ids = enc.pop("offset_mapping").tolist(), enc["input_ids"].tolist()
                h = self.model(**enc.to(self.device), output_hidden_states=True).hidden_states[self.layer]
                h = torch.nn.functional.normalize(h.float(), dim=-1)
                for b in range(len(ids)):
                    k = [j for j, (t, (s, e)) in enumerate(zip(ids[b], offs[b])) if t not in self.skip and e > s]
                    out.append((h[b, k], [tuple(offs[b][j]) for j in k]))
        return out

    @staticmethod
    def align(ea, eb):
        sim = ea @ eb.T
        fwd, bwd = sim.argmax(1).tolist(), sim.argmax(0).tolist()
        return {(i, j): float(sim[i, j]) for i, j in enumerate(fwd) if bwd[j] == i}


def _edge_ok(text, lang):
    if lang not in STOP:
        return True
    w = re.findall(r"[^\W\d_]+", text.lower())
    return bool(w) and w[-1] not in STOP[lang]


def _clause_edges_ok(e_off, e_words, e2m, a2, b2, cov):
    """절 단위 E 구간 바로 앞뒤 확인함. 정렬 안 된 단어가 2개 이상 붙어 있거나,
    가장 가까운 정렬 단어가 M 구간 안쪽을 가리키면 내용 누락·중복 가능성이 있어 제외함."""
    after = [j for j, (a, b) in enumerate(e_off) if a >= b2 and j in e_words]
    before = [j for j, (a, b) in enumerate(e_off) if b <= a2 and j in e_words][::-1]
    for seq, ok in ((after, lambda i: i > cov[-1]), (before, lambda i: i < cov[0])):
        k = next((n for n, j in enumerate(seq) if j in e2m), None)
        if (k is None and len(seq) > 1) or (k is not None and (k > 1 or not ok(e2m[seq[k]]))):
            return False
    return True


def _candidates(mi, m_off, ei, e_off, links, strategy, lo, hi, min_cov=0.7, short_sim=0.8):
    """alignment와 일관된 (M 구간, E 구간) 후보 반환함."""
    m2e, e2m = collections.defaultdict(set), {}
    for (i, j), v in links.items():
        m2e[i].add((j, v))
        e2m[j] = i
    mt, et = mi.text, ei.text
    m_words = {i for i, (a, b) in enumerate(m_off) if re.search(r"\w", mt[a:b])}
    e_words = {i for i, (a, b) in enumerate(e_off) if re.search(r"\w", et[a:b])}
    if strategy == "clause":
        S, E, ES, EE = mi.CS, mi.CE, sorted(ei.CS), ei.CE
    else:
        S, E, ES, EE = mi.S, mi.E, sorted(ei.S), None
    need = min_cov - 0.1 if strategy == "clause" else min_cov
    n, out = len(mt), []
    for s in S:
        for e in E:
            if not (lo * n <= e - s <= hi * n) or e > mi.end:
                continue
            if any(a < s < b or a < e < b for a, b in m_off):
                continue  # token 중간을 자르는 구간 제외함
            cov = [i for i, (a, b) in enumerate(m_off) if a >= s and b <= e and i in m_words]
            if not cov or sum(bool(m2e[i]) for i in cov) / len(cov) < need:
                continue
            if len(cov) <= 2 and any(v < short_sim for i in cov for _, v in m2e[i]):
                continue  # 짧은 구간은 신뢰도 높은 link만 사용함
            js = {j for i in cov for j, _ in m2e[i]}
            a_lo, b_hi = min(e_off[j][0] for j in js), max(e_off[j][1] for j in js)
            if strategy == "clause":
                ends = EE
            elif e in mi.P or mi.lang in ("ko", "ja") or ei.lang in ("ko", "ja"):
                ends = ei.C  # 뒤에 조사가 붙거나 한·일 명사구 삽입이면 E는 내용어로 끝나야 함
            else:
                ends = ei.E
            k = bisect.bisect_right(ES, a_lo) - 1
            b_c = [x for x in ends if x >= b_hi]
            if k < 0 or not b_c:
                continue
            a2, b2 = ES[k], min(b_c)
            inside = [j for j, (a, b) in enumerate(e_off) if a >= a2 and b <= b2 and j in e_words]
            if not inside:
                continue
            bad = [j in e2m and e2m[j] not in cov for j in inside]
            if any(bad[:2] + bad[-2:]) or sum(bad) > (0.1 * len(inside) if strategy == "clause" else 0):
                continue  # E 구간이 M 구간 밖과 정렬되면 일관성 없음 (절 내부만 10% 허용)
            if inside[0] not in e2m or inside[-1] not in e2m or not m2e[cov[0]] or not m2e[cov[-1]]:
                continue  # 양쪽 구간의 양 끝 단어가 서로 정렬돼 있어야 함 (중복·누락 방지)
            if strategy == "clause" and not _clause_edges_ok(e_off, e_words, e2m, a2, b2, cov):
                continue
            if sum(j in e2m for j in inside) / len(inside) < need:
                continue
            if not (0.4 <= len(inside) / len(cov) <= 2.5) or b2 - a2 >= 0.9 * len(et):
                continue
            if strategy == "phrase" and not (mi.is_np(s, e) and ei.is_np(a2, b2)):
                continue
            if not _edge_ok(mt[s:e], mi.lang) or not _edge_ok(et[a2:b2], ei.lang):
                continue
            out.append((s, e, a2, b2))
    return out


def _splice(m_text, parts):
    """M 문장의 구간들을 E 구간으로 바꿈. parts: (s, e, text, lang, E 문장 첫머리 여부, strategy)."""
    t = m_text
    for s, e, x, lang, e_first, _ in sorted(parts, reverse=True):
        if lang in ("en", "es") and x[:1].isalpha():
            if s == 0:
                x = x[0].upper() + x[1:]
            elif e_first and x[:1].isupper() and x[1:2].islower():
                x = x[0].lower() + x[1:]  # E 문장 첫 단어라서 대문자인 경우만 소문자로 바꿈
        t = t[:s] + x + t[e:]
    s, e, x, lang, _, strategy = max(parts, key=lambda p: p[1])
    tail = m_text[e:].strip()
    if strategy == "clause" and tail in ("。", ".") and t.endswith(tail):
        t = t[:-1] + ("。" if lang == "ja" else ".")  # 마지막 절 언어에 맞춰 문장부호 바꿈
    return " ".join(t.split())


def make_mixed(row, infos, encs, aligner, rng, tgt, m, emb, kind, cache=None):
    """주 언어 m 문장에 emb 언어 구간을 끼워 code-switched 입력 생성함. kind=clause면 첫 삽입을 절 단위로 함.
    정답은 target 언어 원문 그대로 씀. emb 중 하나라도 못 끼우면 None 반환함. cache는 row 단위 계산 재사용용."""
    cache = {} if cache is None else cache
    parts, used = [], []
    for k, e_lang in enumerate(emb):
        strategy = "clause" if kind == "clause" and k == 0 else "phrase"
        if (m, e_lang, strategy) not in cache:
            if (m, e_lang) not in cache:
                cache[(m, e_lang)] = aligner.align(encs[m][0], encs[e_lang][0])
            lo, hi = (0.2, 0.7) if strategy == "clause" else (0.04, 0.45)
            cache[(m, e_lang, strategy)] = _candidates(infos[m], encs[m][1], infos[e_lang], encs[e_lang][1],
                                                       cache[(m, e_lang)], strategy, lo, hi)
        cands = [c for c in cache[(m, e_lang, strategy)] if all(c[1] <= u[0] or c[0] >= u[1] for u in used)
                 and sum(u[1] - u[0] for u in used) + c[1] - c[0] <= 0.85 * len(row[m])]
        rng.shuffle(cands)
        pick = next(((s, e, a, b) for s, e, a, b in cands
                     if re.search(SCRIPT[e_lang], row[e_lang][a:b]) and row[e_lang][a:b] != row[m][s:e]), None)
        if pick is None:
            return None  # 끼울 구간이 없거나, 끼워도 실제로 섞이지 않음
        s, e, a, b = pick
        x = row[e_lang][a:b]
        if strategy == "clause" and row[e_lang][b:b + 1] in CLAUSE_P and row[m][e:e + 1] not in PUNCT:
            x += row[e_lang][b]
        used.append((s, e))
        parts.append((s, e, x, e_lang, a == 0, strategy))
    x = _splice(row[m], parts)
    if x in (row[m], row[tgt]) or not 15 <= len(x) <= 220:
        return None
    return {"input": x, "target_language": tgt, "input_type": "mixed", "answer": row[tgt], "matrix": m,
            "embedded": [p[3] for p in parts], "strategy": "+".join(p[5] for p in parts)}


def _alloc(n, props):
    """n을 비율대로 정수 배분함 (최대 나머지 방식)."""
    tot = sum(props.values())
    raw = {k: n * v / tot for k, v in props.items()}
    out = {k: int(x) for k, x in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[:n - sum(out.values())]:
        out[k] += 1
    return out


def _norm(s):
    return re.sub(r"\W+", " ", s.lower()).strip()


def build_train_data(cfg):
    """train/dev JSONL 생성함. single 6방향 + 합성 mixed(en/ko target).
    mixed는 (target, 문자 체계 조합, 절/구 단위) 칸별 할당량을 채우는 방식임. 통계는 *_stats.json에 저장함."""
    c = cfg["train_data"]
    rng = random.Random(c["seed"])
    an = Analyzer()
    pool = load_pool(c["raw_dir"], c["max_en_chars"])
    rng.shuffle(pool)
    stats = Counter(pool_after_rules=len(pool))
    it = iter(pool)

    def next_complete():
        """완결 문장 필터를 통과한 다음 row와 형태소 정보 반환함 (pool 순서대로 소비, 재사용 없음)."""
        for r in it:
            infos = {l: Info(r[l], l, an) for l in LANGS}
            if all(complete(r[l], l, infos[l].ms) for l in LANGS):
                return r, infos
            stats["dropped_incomplete"] += 1
        raise RuntimeError("pool 부족함")

    aligner = Aligner(c["align_model"], c["align_revision"], c["align_layer"])
    banned = {_norm(x) for p in c["exclude_eval"] for r in load_jsonl(p) for x in (r["input"], r.get("answer", "")) if x}
    seen = set()  # dev·train 전체에서 같은 (입력, target) 중복 막음

    def take(out, ex):
        key = (ex["input"], ex["target_language"])
        if key in seen or _norm(ex["input"]) in banned or _norm(ex["answer"]) in banned:
            stats["dropped_dup_or_eval_overlap"] += 1
            return False
        seen.add(key)
        out.append(ex)
        return True

    def build(n_single, n_mixed):
        out = []
        for src, tgt in DIRECTIONS:  # 방향별로 다른 row 사용함
            k = 0
            while k < n_single:
                r, _ = next_complete()
                k += take(out, {"input": r[src], "target_language": tgt, "input_type": "single",
                                "answer": r[tgt], "src_lang": src})
        quota = {}
        for tgt in TARGETS:
            for cat, n_cat in _alloc(n_mixed, c["mix_profile"][tgt]).items():
                n_cl = round(n_cat * c["clause_share"][tgt])
                quota[(tgt, cat, "clause")], quota[(tgt, cat, "phrase")] = n_cl, n_cat - n_cl
        while any(quota.values()):
            try:
                batch = [next_complete() for _ in range(512)]
            except RuntimeError:
                raise RuntimeError(f"pool 부족함. 남은 quota: { {k: v for k, v in quota.items() if v} }")
            encs = aligner.encode([r[l] for r, _ in batch for l in LANGS])
            for i, (r, infos) in enumerate(batch):
                row_encs, cache, done = {l: encs[4 * i + j] for j, l in enumerate(LANGS)}, {}, set()
                for _ in range(c["tries_per_row"]):
                    cells = [q for q, v in quota.items() if v > 0 and q[0] not in done]  # row당 target별 최대 1개
                    if not cells:
                        break
                    tgt, cat, kind = rng.choices(cells, weights=[quota[q] for q in cells])[0]  # 많이 남은 칸 우선
                    m, emb = rng.choice(COMBOS[cat])
                    emb = rng.sample(emb, len(emb))  # 절 단위는 첫 삽입 언어에 적용하므로 순서 섞음
                    stats["mixed_tried"] += 1
                    ex = make_mixed(r, infos, row_encs, aligner, rng, tgt, m, emb, kind, cache)
                    if (ex and script_cat(ex["input"]) == cat and ("clause" in ex["strategy"]) == (kind == "clause")
                            and take(out, ex)):
                        quota[(tgt, cat, kind)] -= 1
                        stats["mixed_made"] += 1
                        done.add(tgt)
        return out

    dev = build(c["dev_single_per_direction"], c["dev_mixed_per_target"])
    train = build(c["single_per_direction"], c["mixed_per_target"])
    for name, rows in (("dev", dev), ("train", train)):
        rng.shuffle(rows)
        rows = [{"id": i, **r} for i, r in enumerate(rows)]
        check_eval_rows(rows, need_answer=True)
        save_jsonl(rows, c[f"{name}_out"])
        stats[f"{name}_rows"] = len(rows)
        stats.update({f"{name}_{t}_{l}": v for (t, l), v in Counter((r["input_type"], r["target_language"]) for r in rows).items()})
        mixed = [r for r in rows if r["input_type"] == "mixed"]
        stats.update({f"{name}_mixed_{t}_{cat}_{k}": v for (t, cat, k), v in Counter(
            (r["target_language"], script_cat(r["input"]), "clause" if "clause" in r["strategy"] else "phrase")
            for r in mixed).items()})
    stats["mixed_success_rate"] = round(stats["mixed_made"] / stats["mixed_tried"], 3)
    with open(c["train_out"].replace(".jsonl", "_stats.json"), "w", encoding="utf-8") as f:
        json.dump({"config": c, "stats": dict(stats)}, f, ensure_ascii=False, indent=2)
    return dict(stats)
