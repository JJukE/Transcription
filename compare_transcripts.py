"""Compare local ASR outputs. Differences are observations, not accuracy scores."""
import argparse
import collections
import csv
import difflib
import hashlib
import json
import re
import unicodedata
from pathlib import Path


def normalize(text):
    value = unicodedata.normalize('NFKC', text).casefold()
    return ''.join(c for c in value if c.isalnum())


def window_text(result, start, end):
    fragments = []
    for segment in result['segments']:
        if segment.get('words'):
            for word in segment['words']:
                if start <= (word['start']+word['end'])/2 < end:
                    fragments.append(word['word'])
        elif start <= (segment['start']+segment['end'])/2 < end:
            fragments.append(segment['text'])
    return ''.join(fragments).strip()


def time_label(seconds):
    seconds = round(seconds)
    return f'{seconds//60:02}:{seconds%60:02}'


def changes(left, right):
    return [{'operation':op,'whisper':left[a:b],'qwen':right[c:d]}
            for op,a,b,c,d in difflib.SequenceMatcher(None,left,right,autojunk=False).get_opcodes() if op!='equal']


def audit(result, duration):
    segments = result['segments']
    errors, overlaps, gaps = [], [], []
    previous_start, previous_end = -1., 0.
    for i, segment in enumerate(segments):
        start, end = segment['start'], segment['end']
        if not (0 <= start <= end <= duration + 0.05) or start < previous_start:
            errors.append(i)
        if start < previous_end - 1e-4:
            overlaps.append({'segment':i,'seconds':previous_end-start,'start':start})
        if start-previous_end >= 5:
            gaps.append({'start':previous_end,'end':start,'seconds':start-previous_end})
        previous_start, previous_end = start, max(previous_end,end)
        for word in segment.get('words',[]):
            if not 0<=word['start']<=word['end']<=duration+0.05:
                errors.append(f'word_in_segment_{i}')
    if duration-previous_end>=5:
        gaps.append({'start':previous_end,'end':duration,'seconds':duration-previous_end})
    consecutive = []
    for i in range(1,len(segments)):
        if normalize(segments[i]['text']) and normalize(segments[i]['text'])==normalize(segments[i-1]['text']):
            consecutive.append({'start':segments[i]['start'],'end':segments[i]['end']})
    return {'segment_count':len(segments),'word_count':sum(len(s.get('words',[])) for s in segments),
            'text_characters':len(result['text']),'invalid_timestamps':errors,
            'overlapping_segments':overlaps,'gaps_at_least_5s':gaps,'consecutive_equal_segments':consecutive,
            'last_segment_end':previous_end}


def load_terms(path=None):
    if path is None:
        return []
    terms = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(terms, list) or any(not isinstance(t, str) or not t.strip() for t in terms):
        raise ValueError("--terms-file must contain a JSON array of nonempty strings")
    return list(dict.fromkeys(terms))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--whisper',required=True)
    parser.add_argument('--qwen',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--terms-file', help='Optional local JSON array of domain terms; default: no term list')
    args=parser.parse_args()
    wp,qp=Path(args.whisper),Path(args.qwen)
    whisper,qwen=json.loads(wp.read_text()),json.loads(qp.read_text())
    terms=load_terms(args.terms_file)
    duration=qwen['duration'];out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    negatives=re.compile(r'안\s|않|없|아니|못|불가능|not\b|never\b|without\b',re.I)
    numbers=re.compile(r'\d+(?:[.,]\d+)*(?:%|배|개|차|초|분|시간|GB|MB)?',re.I)
    rows=[]
    for start in range(0,int(duration)+1,60):
        end=min(start+60,duration)
        if end<=start:continue
        w,q=window_text(whisper,start,end),window_text(qwen,start,end)
        wn,qn=normalize(w),normalize(q)
        numeric_w,numeric_q=numbers.findall(w),numbers.findall(q)
        neg_w,neg_q=negatives.findall(w),negatives.findall(q)
        term_w={t:len(re.findall(re.escape(t),w,re.I)) for t in terms if re.search(re.escape(t),w,re.I)}
        term_q={t:len(re.findall(re.escape(t),q,re.I)) for t in terms if re.search(re.escape(t),q,re.I)}
        flags=[]
        if bool(wn)!=bool(qn):flags.append('one_model_empty')
        if numeric_w!=numeric_q:flags.append('numeric_rendering_differs')
        if len(neg_w)!=len(neg_q):flags.append('negation_expression_count_differs')
        if term_w!=term_q:flags.append('english_term_rendering_differs')
        if min(len(wn),len(qn)) and max(len(wn),len(qn))>1.5*min(len(wn),len(qn)):
            flags.append('large_text_length_difference')
        rows.append({'start':start,'end':end,'whisper':w,'qwen':q,'flags':flags,
                     'normalized_text_equal':wn==qn,'raw_changes':changes(w,q),
                     'normalized_changes':changes(wn,qn),'numbers':{'whisper':numeric_w,'qwen':numeric_q},
                     'english_terms':{'whisper':term_w,'qwen':term_q},
                     'negation_expression_counts':{'whisper':len(neg_w),'qwen':len(neg_q)}})
    summary={'duration_seconds':duration,'windows':len(rows),'normalized_equal_windows':sum(r['normalized_text_equal'] for r in rows),
             'flag_counts':dict(collections.Counter(f for r in rows for f in r['flags'])),
             'whisper_audit':audit(whisper,duration),'qwen_audit':audit(qwen,duration),
             'window_assignment':{'whisper':'word midpoint when available, otherwise segment midpoint','qwen':'word midpoint'},
             'source_sha256':{str(wp):hashlib.sha256(wp.read_bytes()).hexdigest(),str(qp):hashlib.sha256(qp.read_bytes()).hexdigest()},
             'limitations':['No human reference transcript; no CER/WER or accuracy estimate.',
                            'Flags indicate transcript disagreement, not which model is correct.',
                            'Silence/overlap/segment differences do not prove missing speech.',
                            'Numbers written as Korean words and English terms written in Hangul need manual review.',
                            'Whisper full-run timing is unavailable; speed superiority is not established.']}
    payload={'summary':summary,'windows':rows,'qualitative_review_status':'pending_content_review'}
    (out/'analysis.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
    with (out/'comparison_60s.tsv').open('w',newline='') as file:
        writer=csv.writer(file,delimiter='\t');writer.writerow(['start_seconds','end_seconds','whisper_large_v3','qwen3_asr_1_7b','review_candidates'])
        for r in rows:writer.writerow([r['start'],r['end'],r['whisper'],r['qwen'],','.join(r['flags'])])
    for kind in ['raw','normalized']:
        sections=[]
        for row in rows:
            sections.append(f"## {time_label(row['start'])}–{time_label(row['end'])}\n")
            for change in row[f'{kind}_changes']:
                sections.append(f"- Whisper: {change['whisper']}\n+ Qwen: {change['qwen']}\n")
        (out/f'diff_{kind}.txt').write_text('\n'.join(sections))
    receipt_path=qp.with_name(qp.stem+'.receipt.json')
    receipt=json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
    elapsed=receipt.get('elapsed_seconds_including_model_load')
    elapsed_label=f'{elapsed:.2f}초' if elapsed is not None else '미제공'
    allocated=receipt.get('peak_cuda_allocated_bytes')
    reserved=receipt.get('peak_cuda_reserved_bytes')
    memory_label=f'{allocated/2**30:.2f} / {reserved/2**30:.2f} GiB' if allocated is not None and reserved is not None else '미제공'
    draft=f'''# Qwen3-ASR-1.7B / Whisper large-v3 미팅 전사 비교

전사 JSON: {wp.name} / {qp.name}, {duration:.3f}초. 제공된 두 전사 결과를 비교하며 원본 파일은 보존한다.

## 실행 및 구조 검증

- Qwen 처리 시간(모델 로딩 포함): {elapsed_label}.
- PyTorch 최대 GPU 할당/예약 메모리: {memory_label}.
- Whisper / Qwen 세그먼트 수: {summary['whisper_audit']['segment_count']} / {summary['qwen_audit']['segment_count']}. 자막 구성 방식이 달라 개수는 품질 지표가 아니다.
- Qwen 타임스탬프 오류: {len(summary['qwen_audit']['invalid_timestamps'])}개.
- 60초 비교 구간: {len(rows)}개. 단어가 제공되면 단어 중심 시각, 그렇지 않으면 세그먼트 중심 시각으로 구간에 배정했다.
- 입력·모델·코드 해시와 실행 설정은 전사 receipt에, 네트워크 차단/복원은 network.receipt.json에 기록했다.

## 기계적으로 발견한 검토 후보

```json
{json.dumps(summary['flag_counts'],ensure_ascii=False,indent=2)}
```

이 항목들은 두 출력의 표현 차이이며 오류 판정이 아니다. 숫자 한글 표기, 전문용어 음역, 청크/타임스탬프 배정 차이도 후보를 만든다. 공백·문장부호·대소문자를 정규화한 diff를 함께 생성했다.

## 내용 검토

전사 내용 검토 대기 중. 전문용어·수치·부정 표현·누락/반복·의미 차이에 대한 정성 판단은 추가 검토 후 기록한다.

## 해석의 한계

정답 전사문이 없으므로 CER/WER나 정확도를 산출하지 않는다. 기존 review_pass.json은 다른 모델의 보조 전사이며 정답이 아니다. 기존 Whisper 전체 실행 시간이 없어 속도 우열도 단정하지 않는다. 무음과 자막 분할 차이를 발화 누락으로 단정하지 않는다.

## 파일

- comparison_60s.tsv: 전 구간 나란히 보기.
- analysis.json: 구간별 표현 차이와 검토 후보, 구조 검증.
- diff_raw.txt / diff_normalized.txt: 원문/정규화 표현 diff.
- {qp.stem}.*: Qwen 전사, 자막, diagnostics, 실행 receipt.
'''
    (out/'comparison_report.md').write_text(draft)
    # Print only aggregate metadata, never meeting text.
    print(json.dumps({k:summary[k] for k in ['duration_seconds','windows','normalized_equal_windows','flag_counts']},ensure_ascii=False))


if __name__=='__main__':main()
