import pandas as pd

from bokeh.plotting import figure, show, output_notebook, output_file
from bokeh.tile_providers import get_provider, Vendors
from bokeh.models import ColumnDataSource, LabelSet, Select, CDSView, IndexFilter
from bokeh.layouts import column, row
from bokeh.transform import dodge

from xlib import lon8lat_to_mercator

X_WEST, Y_NORTH = lon8lat_to_mercator(100, 50)
X_EAST, Y_SOUTH = lon8lat_to_mercator(135, 20)
RANK_DATA_PATH = '../../data/Rank.csv'
CITIES_PATH = '../../data/world_cities.csv'
OUTPUT_PATH = 'univ_rank.html'

output_notebook()
output_file(OUTPUT_PATH)

## Read and process data

rank_data = pd.read_csv(RANK_DATA_PATH)
cities = pd.read_csv(CITIES_PATH, index_col='name')
rank_data

rank_data['Location'] = rank_data['Location'].map(lambda s:s.split(',')[0])
rank_data['Location'] = rank_data['Location'].map(lambda s:'Xi’an' if s == 'Xi\'an' else s)
rank_data['Location'] = rank_data['Location'].map(lambda s:'Tianjin' if s == 'Nankai' else s)
rank_data['GlobalRanking'] = rank_data['GlobalRanking'].map(lambda s:'No.' + str(s))

total_data = rank_data.join(cities, on='Location')
total_data['mercator_x'], total_data['mercator_y'] = \
    zip(*total_data.apply(lambda r:lon8lat_to_mercator(r['lng'], r['lat']), axis=1))

overall_data = total_data[['SchoolName', 'GlobalRanking', 'GlobalScores','mercator_x', 'mercator_y']].drop_duplicates()
overall_data = overall_data.iloc[::-1]

overall_data

overall_data_dict = overall_data.to_dict('list')

source = ColumnDataSource(
    data=dict(
        school=overall_data_dict['SchoolName'],
        school_rank=overall_data_dict['GlobalRanking'],
        school_score=overall_data_dict['GlobalScores'],
        lon=overall_data_dict['mercator_x'],
        lat=overall_data_dict['mercator_y'],
        subject_score = [0]*20,  # init
        subject_score_str = ['']*20,  # init
        subject_rank = ['']*20,  # init
             )
)

subjects = list(total_data['SubjectName'].drop_duplicates())
subjects

len(subjects)

schools = list(overall_data['SchoolName'].drop_duplicates())
schools

source_compare = ColumnDataSource(
    data=dict(
        subjects=subjects,
        school1_subjects_score=[],
        school2_subjects_score=[],
        school1_subjects_rank=[],
        school2_subjects_rank=[],
        school1_subjects_scorestr=[],
        school2_subjects_scorestr=[],
             )
)

## Create map

tile_provider = get_provider(Vendors.CARTODBPOSITRON_RETINA)

univ_map = figure(x_range=(X_WEST, X_EAST),
                  y_range=(Y_NORTH, Y_SOUTH),
                  x_axis_type="mercator", y_axis_type="mercator",
                  tools='box_select',) # toolbar_location='below'
univ_map.add_tile(tile_provider)

univ_map.circle(source=source, x="lon", y="lat",
                size=10, alpha=0.8, color='crimson')

## Create university rank glyph

total_rank = figure(y_range=overall_data_dict['SchoolName'], x_range=(0, 100),
                    tools='tap', #toolbar_location='below',
                    title="Global Rank (use SHIFT to select MULTIPLE target)",
                    ) #　y_axis_location="right", width=600
total_rank.hbar(source=source, y='school',
                right='school_score', height=0.5, color='crimson', alpha=0.5)

label_score = LabelSet(x='school_score', y='school', text='school_score',
        x_offset=-35, y_offset=-10, source=source)
label_rank = LabelSet(x=None, y='school', text='school_rank',
        x_offset=+10, y_offset=-10, source=source)

total_rank.add_layout(label_score)
total_rank.add_layout(label_rank)

## Create subject view, compare all universities

def ticker_subject_change(attrname, old, new):
    update_subject(new)

    
def update_subject(selected):
    new_scores = []
    new_ranks = []
    new_scores_str = []
    for school in schools:
        if school == 'Xi\'an Jiaotong University':
            school = 'Xi\\\'an Jiaotong University'
        ret = total_data.query(f'SchoolName == \'{school}\' and SubjectName == \'{selected}\'')
        if len(ret) == 0:
            new_scores.append(0)
            new_ranks.append('')
            new_scores_str.append('')
        else:
            assert len(ret) == 1
            new_scores.append(float(list(ret['SubjectScores'])[0]))
            new_ranks.append(f"No.{list(ret['SubjectRanking'])[0]}")
            new_scores_str.append('{:.1f}'.format(float(list(ret['SubjectScores'])[0])))
            
    source.data['subject_score'] = new_scores
    source.data['subject_rank'] = new_ranks
    source.data['subject_score_str'] = new_scores_str
    
#     ids = sorted(range(len(new_scores)), key=lambda k: new_scores[k])
#     ids = list(filter(lambda ix: new_scores[ix] > 0, ids))
  

default_subject = 'Computer Science'
ticker_subject = Select(title='subject', value=default_subject, options=subjects)
ticker_subject.on_change('value', ticker_subject_change)
update_subject(default_subject)

subject_rank = figure(y_range=schools, x_range=(0, 100),  title="Subject Rank (use SHIFT to select MULTIPLE target)", # y_range=schools, 
                      tools='tap', ) # width=350, toolbar_location='below', 

subject_rank.hbar(y='school', right='subject_score', source=source, #view=subject_view,
       height=.5, color='crimson', alpha=0.5)

# subject_rank.y_range.range_padding = 0.1
# subject_rank.ygrid.grid_line_color = None
# subject_rank.yaxis.visible = False
# subject_rank.legend.location = "bottom_right"
# subject_rank.legend.orientation = "horizontal"

label_subject_score = LabelSet(x='subject_score_str', y='school', text='subject_score_str',
        x_offset=-35, y_offset=-10, source=source)
label_subject_rank = LabelSet(x=None, y='school', text='subject_rank',
        x_offset=+10, y_offset=-10, source=source)

subject_rank.add_layout(label_subject_score)
subject_rank.add_layout(label_subject_rank)

subject_rank_view = column(ticker_subject, subject_rank, )

## School compare

def ticker_school1_change(attrname, old, new):
    update_school(1, new)

    
def ticker_school2_change(attrname, old, new):
    update_school(2, new)

    
def update_school(IX, selected):
    if selected == 'Xi\'an Jiaotong University':
        selected = 'Xi\\\'an Jiaotong University'
        
    if IX==1:
        score_name = 'school1_subjects_score'
        rank_name = 'school1_subjects_rank'
        scorestr_name = 'school1_subjects_scorestr'
    else:
        score_name = 'school2_subjects_score'
        rank_name = 'school2_subjects_rank'
        scorestr_name = 'school2_subjects_scorestr'
        
    new_scores = []
    new_ranks = []
    new_scores_str = []
    for subject in subjects:
        ret = total_data.query(f'SchoolName == \'{selected}\' and SubjectName == \'{subject}\'')
        if len(ret) == 0:
            new_scores.append(0)
            new_ranks.append('')
            new_scores_str.append('')
        else:
            assert len(ret) == 1
            new_scores.append(float(list(ret['SubjectScores'])[0]))
            new_ranks.append(f"No.{ret['SubjectRanking'].to_list()[0]}")
            new_scores_str.append('{:.1f}'.format(float(list(ret['SubjectScores'])[0])))
            
    source_compare.data[score_name] = new_scores
    source_compare.data[rank_name] = new_ranks
    source_compare.data[scorestr_name] = new_scores_str
    
#     ids = sorted(range(len(new_scores)), key=lambda k: new_scores[k])
#     ids = list(filter(lambda ix: new_scores[ix] > 0, ids))
  

default_school1 = 'Peking University'
default_school2 = 'Tsinghua University'
ticker_school1 = Select(title='School1', value=default_school1, options=schools)
ticker_school2 = Select(title='School2', value=default_school2, options=schools)
ticker_school1.on_change('value', ticker_school1_change)
ticker_school2.on_change('value', ticker_school2_change)
update_school(1, default_school1)
update_school(2, default_school2)

school_compare = figure(y_range=subjects, x_range=(0, 100), title="School1 V.S. School2",
           tools="", height=900) # toolbar_location=None, 

school_compare.hbar(y=dodge('subjects', +.2, range=school_compare.y_range),
                    right='school1_subjects_score', source=source_compare,
                    height=0.4, color="skyblue", legend_label="School1", alpha=0.5)
school_compare.hbar(y=dodge('subjects', -.2, range=school_compare.y_range),
                    right='school2_subjects_score', source=source_compare,
                    height=0.4, color="navy", legend_label="School2", alpha=0.5)

school_compare.legend.location = "top_right"
school_compare.legend.orientation = "vertical"

label_subject_score1 = LabelSet(x='school1_subjects_score', y='subjects', text='school1_subjects_scorestr',
        x_offset=-35, y_offset=0, source=source_compare)
label_subject_rank1 = LabelSet(x=None, y='subjects', text='school1_subjects_rank',
        x_offset=+10, y_offset=0, source=source_compare)
label_subject_score2 = LabelSet(x='school2_subjects_score', y='subjects', text='school2_subjects_scorestr',
        x_offset=-35, y_offset=-20, source=source_compare)
label_subject_rank2 = LabelSet(x=None, y='subjects', text='school2_subjects_rank',
        x_offset=+10, y_offset=-20, source=source_compare)

school_compare.add_layout(label_subject_score1)
school_compare.add_layout(label_subject_rank1)
school_compare.add_layout(label_subject_score2)
school_compare.add_layout(label_subject_rank2)

school_compare_view = column(row(ticker_school1, ticker_school2), school_compare)

## Display

layout_1 = row(total_rank, univ_map)
layout_2 = row(subject_rank_view, school_compare_view)
layout = column(layout_1, layout_2)
show(layout)




from bokeh.io import curdoc
curdoc().add_root(layout)
curdoc().title = "Demo"