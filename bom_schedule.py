import streamlit as st
import pandas as pd
import numpy as np
import requests
# import matplotlib.pyplot as plt
# import seaborn as sns

BOM_URL = 'https://raw.githubusercontent.com/bcbooks/scriptures-json/master/book-of-mormon.json'

st.title('Book of Mormon Daily Scripture Study Schedule')

@st.cache_data
def load_data():
    data = requests.get(BOM_URL).json()
    df = pd.json_normalize(data,
                           record_path=['books','chapters','verses'],
                           meta=[
                               ['books','book'],
                               ['books','chapters','reference'],
                               ['books','chapters','chapter'],
                            ])\
        .rename(columns={
            'books.book':'book',
            'books.chapters.reference': 'chapter',
            'books.chapters.chapter': 'chapter_num',
        })
    df['word_count'] = df.text.str.strip().str.split(r'\s+',regex=True).apply(len)
    df['chapter_idx'] = df[['chapter']].join(df.chapter.drop_duplicates().reset_index(drop=True).reset_index().set_index('chapter'), on='chapter')['index']
    
    g = df.groupby('chapter_idx')
    chapters = pd.DataFrame({
        'idx': g.chapter_idx.first(),
        'book': g.book.first(),
        'chapter': g.chapter.first(),
        'chapter_num': g.chapter_num.first(),
        'versus': g.verse.nunique(),
        'word_count': g.word_count.sum(),
    })

    # Remove chapter number on single chapter books
    single_chapter_book = (chapters.groupby('book').chapter.nunique() == 1).rename('single_chapter_book')
    single_chapter_book = chapters.join(single_chapter_book, on='book').single_chapter_book
    chapters['chapter'] = chapters.book.where(single_chapter_book, chapters.chapter)

    assert chapters.chapter.nunique() == len(chapters)

    return chapters

## Load the Data
data_load_state = st.text('Loading data...')
chapters = load_data()
data_load_state.text('')


## Prompt Inputs
columns = iter(st.columns(2))
with next(columns):
    CHAPTER = st.text_input(label='Starting Chapter', value='1 Nephi 1')
with next(columns):
    DAYS = st.number_input(label='Days Left', min_value=1, value=240)

## Filter the chapters down to be after the starting chapter
chapter_selection = (
    (chapters.chapter.str.lower() == CHAPTER.lower())
    | (chapters.book.str.lower() == CHAPTER.lower())
)
if not chapter_selection.any():
    st.warning('Unknown Chapter')
    st.stop()
chapters = chapters.loc[chapter_selection.idxmax():]

TARGET = chapters.word_count.sum() / DAYS

def calculate_num_days_per_chapter(chapters, TARGET):
    def calc_error(wc, days=1):
        return (((wc / days) - TARGET) ** 2) * days

    current_error = calc_error(chapters.word_count)
    parts = chapters.word_count // TARGET
    split_low = calc_error(chapters.word_count, parts)
    split_high = calc_error(chapters.word_count, parts + 1)
    is_lower = (split_low < current_error) | (split_high < current_error)
    return parts.where(split_low < split_high, parts + 1).where(is_lower, 1).astype(int)

def calculate_schedule(chapters, TARGET):
    days = calculate_num_days_per_chapter(chapters, TARGET)

    schedule = pd.DataFrame({
        'idx': chapters.idx,
        'days': days,
        'chapters': 1,
        'wc': chapters.word_count // days,
        'versus': chapters.versus / days,
    })
    schedule['err'] = ((schedule.wc - TARGET) ** 2) * schedule.days

    # Calculate the consecutive days that would benefit the most
    # from being combined. And greedily combine them until there
    # are no more consecutive days that would benefit from combining
    while True:
        nxt = schedule.shift(-1)
        combined_error = (((schedule.wc + nxt.wc) - TARGET) ** 2)
        separate_error = schedule.err + nxt.err
        does_not_combine_multi_days = (schedule.days + nxt.days) == 2
        potential = (combined_error - separate_error).where(does_not_combine_multi_days, np.nan)
        if potential.min() >= 0:
            break
        i = potential.idxmin()
        # Absorb the next line
        schedule.loc[i, ['chapters', 'wc', 'versus', 'err']] = (
            schedule.loc[i].chapters + nxt.loc[i].chapters,
            schedule.loc[i].wc + nxt.loc[i].wc,
            schedule.loc[i].versus + nxt.loc[i].versus,
            combined_error.loc[i]
        )
        # Drop the absorbed line
        schedule.drop(index=nxt.loc[i].idx, inplace=True)
    schedule = schedule.reset_index(drop=True)
    assert ((schedule.days > 1) & (schedule.chapters > 1)).any() == False, 'I don\'t want multiple chapters across multiple days'
    return schedule

def metric_row(metrics):
    columns = st.columns(len(metrics))
    for col, (label, value) in zip(columns, metrics.items()):
        with col:
            st.metric(label=label, value=value)

schedule = calculate_schedule(chapters, TARGET)
AVG_WORDS_PER_VERSE = chapters.word_count.sum() / chapters.versus.sum()
schedule['avg_versus'] = (schedule.wc / AVG_WORDS_PER_VERSE).round().astype(int)

schedule = schedule.join(chapters[['book','chapter','chapter_num']].astype(str), on='idx')
schedule = schedule.join(chapters[['book','chapter','chapter_num']].astype(str).add_prefix('end_'), on=schedule.idx + schedule.chapters - 1)
schedule['chapter_part'] = schedule.days.apply(lambda days: list(range(days)))
schedule = schedule.explode('chapter_part').reset_index(drop=True)
schedule['start_verse'] = ((schedule.chapter_part * schedule.versus).astype(float).round() + 1).astype(int).astype(str)
schedule['end_verse'] = ((schedule.chapter_part + 1) * schedule.versus).astype(float).round().astype(int).astype(str)

multi_day = schedule.days > 1
multi_chapter = schedule.chapters > 1
schedule['name'] = schedule.chapter
schedule.loc[multi_chapter, 'name'] = schedule.chapter + '-' + schedule.end_chapter_num
schedule.loc[multi_chapter & (schedule.book != schedule.end_book), 'name'] = schedule.chapter + ' - ' + schedule.end_chapter
schedule.loc[multi_day, 'name'] = schedule.chapter + ':' + schedule.start_verse + '-' + schedule.end_verse

def strip_book_name(r):
    return r['name'][len(r.book):].strip()
continuing_book = schedule.book == schedule.shift().end_book
schedule['short_name'] = schedule.apply(strip_book_name, axis=1).where(continuing_book, schedule.name)

metric_row({
    'Chapters Left': len(chapters),
    'Schedule Days': len(schedule),
    'Versus per Day': '{:.1f}'.format(schedule.avg_versus.mean()),
    'Chapters per Day': '{:.2f}'.format((schedule.chapters / schedule.days).mean()),
})

# fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(8, 2))
# sns.histplot(data=schedule, y='avg_versus', kde=True, ax=ax1)
# sns.scatterplot(data=schedule.reset_index().rename(columns={'index':'day'}), x='day', y='avg_versus', ax=ax2)
# st.write(fig)

st.subheader('Schedule')
st.dataframe(pd.DataFrame({
    'Scripture': schedule['name'],
    '# Versus': schedule.end_verse.astype(int) - schedule.start_verse.astype(int) + 1
}, index=schedule.index.rename('Day')), use_container_width=True)