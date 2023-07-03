import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
sns.set_theme(style='darkgrid')
df = pd.read_csv('/Users/erikrice/Downloads/Presidential Election Results - Sheet1.csv')
#presidential and vice-presidential candidate ages and net ages at the time of the election calculated in Excel.

#cleaning presidential data
print(df['P_Winner_Net_Age'].isna())
p_net_age = df['P_Winner_Net_Age'].dropna()

#have more presidents been older or younger than their opponents?
Older_or_Younger = []
for x in df['P_Winner_Net_Age']:
    if x > 0: Older_or_Younger.append('Older')
    elif x == 0: Older_or_Younger.append('Same Age')
    elif x < 0: Older_or_Younger.append('Younger')
    else: Older_or_Younger.append('n/a')
df['Older_or_Younger'] = Older_or_Younger
f, ax = plt.subplots(figsize = (6,6))
sns.set_palette("Set2")
sns.countplot(data=df[-(df.Older_or_Younger == 'n/a')], x='Older_or_Younger').set(title='Comparative US Presidential Ages Throughout History', xlabel='Relative Age')
plt.show()

#shown as a percentage
print((df['Older_or_Younger'].value_counts(normalize=True))*2)
Relative_Ages = 'Older', 'Younger', 'Same_Age'
Rel_P_Age_Percent = [56.14, 40.35, 3.5]
fig1, ax1 = plt.subplots()
ax1.pie(Rel_P_Age_Percent, labels=Relative_Ages,autopct='%1.1f%%')
ax1.axis('equal')  
plt.title('Comparative Age of US Presidents by Percent', pad=20)
plt.show()

#has this changed over time? adding a time period column
Time_Period = []
for x in df['Election_Year']:
    if x < 1800: Time_Period.append('Late 18th Century')
    elif x < 1850: Time_Period.append('Early 19th Century')
    elif x < 1900: Time_Period.append('Late 19th Century')
    elif x < 1950: Time_Period.append('Early 20th Century')
    elif x < 2000: Time_Period.append('Late 20th Century')
    else: Time_Period.append('Early 21st Century')
df['Time_Period'] = Time_Period

#isolating winners
Winners = df[df['Result'] == 'W']
Losers = df[df['Result'] == 'L']

Time_Period_Result = Winners.groupby('Time_Period')['Older_or_Younger'].value_counts()
print(Time_Period_Result)

#graphing the time period data 
x1 = 'Late 18th Century', 'Early 19th Century', 'Late 19th Century', 'Early 20th Century', 'Late 20th Century', 'Early 21st Century'
y1 = [1, 6, 5, 10, 7, 3]
y2 = [0, 5, 7, 3, 5, 3]
y3 = [0, 1, 0, 0, 0, 0]
recent_history_bar = pd.DataFrame(np.c_[y1, y2, y3], index=x1)
recent_history_bar.plot.bar(color=['green', 'blue', 'cyan']).set(title='Relative Presidential Age by Time Period', xlabel='Time Period', ylabel='# Presidents')
plt.legend(['Older', 'Younger', 'Same Age'])
plt.show()

#visualizing president's age over time. have they really been getting older as a trend?
sns.lineplot(data=Winners, x='Election_Year', y='P_Age').set(title='Presidential Age Over History', xlabel='Election Year', ylabel='Age of President')
sns.set_theme(style='darkgrid')
sns.set_palette("Set2")
plt.show()

#trying out the sunburst plot
time_period_fig = px.sunburst(Winners, path=['Time_Period', 'Party', 'P_Candidate','Older_or_Younger'], values='P_Age', color='P_Age')
time_period_fig.show()

#what about recent American history? looking at the last 50 years. 
Recent_Winners = Winners[Winners['Election_Year'] > 1973]
Recent_Winners_Count = Recent_Winners['Older_or_Younger'].value_counts()
print(Recent_Winners_Count)

#visualizing recent history - bar chart
Recent_Elections_To_Chart = {'Relative Age': ['Older', 'Younger'],
                             'Wins':[6, 6]}
sns.barplot(data=Recent_Elections_To_Chart, x='Relative Age', y='Wins').set(title='Comparative Age of Recent Presidents', xlabel='Relative Age', ylabel='# Presidents')
plt.show()

#pie chart
Relative_Age1 = 'Older', 'Younger'
Wins = [6,6]
fig1, ax1 = plt.subplots()
ax1.pie(Wins, labels=Relative_Age1,autopct='%1.1f%%')
ax1.axis('equal')  
plt.title('Comparative Age of US Presidents in Recent History', pad=20)
plt.show()

#what about vice presidents?
print(df['VP_Winner_Net_Age'].isna())
vp_net_age = df['VP_Winner_Net_Age'].dropna()
VP_Older_or_Younger = []
for x in df['VP_Winner_Net_Age']:
    if x > 0: VP_Older_or_Younger.append('Older')
    elif x == 0: VP_Older_or_Younger.append('Same Age')
    elif x < 0: VP_Older_or_Younger.append('Younger')
    else: VP_Older_or_Younger.append('n/a')
df['VP_Older_or_Younger'] = VP_Older_or_Younger
sns.countplot(data=df[-(df.VP_Older_or_Younger == 'n/a')], x='VP_Older_or_Younger')
plt.show()

#shown as a percentage
print((df['VP_Older_or_Younger'].value_counts(normalize=True))*2)
Relative_VP_Ages = 'Older', 'Younger', 'Same Age'
Rel_VP_Age_Percent = [42.35, 51.62, 6.03]
fig1, ax1 = plt.subplots()
ax1.pie(Rel_VP_Age_Percent, labels=Relative_VP_Ages,autopct='%1.1f%%')
ax1.axis('equal')  
plt.show()

#what about elections with larger age gaps?
Sig_Older = df['P_Winner_Net_Age'] >= 10
print(Sig_Older)

#THE LAST THING I WANT TO DO WITH THIS PROJECT BEFORE CLEANING UP THE FIGURES FOR THE FINAL PROJECT IS LOOK AT LARGER AGE GAPS, WHICH I'VE JUST STARTED HERE. 

#what about the ticket as a whole? adding a column with the mean age of the entire ticket. 
df['Whole_Ticket_Mean_Age'] = df['Election_Year'] - ((df['P_Birth_Year'] + df['VP_Birth_Year']) / 2)

#recreating winners and losers groups w/ this new column. 
Winners = df[df['Result'] == 'W']
Losers = df[df['Result'] == 'L']

#comparing averages
print(Winners['Whole_Ticket_Mean_Age'].mean())
print(Losers['Whole_Ticket_Mean_Age'].mean())

#charting this too. They're all 55 years of age. 
Whole_Ticket_Average = {'Result': ['Winner', 'Loser'],
                        'Mean_Age':[55.68, 55.33]}
sns.barplot(data=Whole_Ticket_Average, x='Result', y='Mean_Age')
plt.show()

#switching gears to the economy now, and inflation. 
#importing historical inflation data
reference_inflation_df = pd.read_csv('/Users/erikrice/Downloads/Inflation Dataset - Sheet1 (3).csv')
df = pd.read_csv('/Users/erikrice/Downloads/Presidential Election Results - Sheet1 (2).csv')
print(reference_inflation_df.head())

#merging the tables
df = df.merge(reference_inflation_df, left_on='Election_Year', right_on='Year', how='outer')

#cleaning the table
df.drop('Year', axis=1, inplace=True)
df['US_Inflation'] = df['United States of America']
df['US_Inflation'] = df['US_Inflation'].str.replace(' %', '')
df['European_Inflation'] = df['Ø EU']
df['Global_Inflation'] = df['Ø World']      
df.drop(['United States of America','Ø EU', 'Ø World'], axis=1, inplace=True)           
print(df.head())

#isolating inflation results
inflation_results = df[['Election_Year', 'Result', 'Electoral_Votes', 'Vote_Percent', 'Popular_Vote', 'Popular_Vote_Percent', 'US_Inflation', 'Hypothetical']]
inflation_results['US_Inflation'] = inflation_results['US_Inflation'].astype(float)
print(inflation_results.dtypes)
inflation_results['US_Inflation'] = inflation_results['US_Inflation'].replace(' %', '')

#visualizing historical inflation data
sns.lineplot(data=inflation_results, x='Election_Year', y='US_Inflation', palette='blue', label='Inflation Rate')
sns.set_theme(style='darkgrid')
plt.show()

#adding 2024 projection to visualization
sns.lineplot(data=inflation_results, x='Election_Year', y='US_Inflation', palette='blue', label='Inflation Rate')
sns.lineplot(data=inflation_results, x='Election_Year', y='Hypothetical', palette='cyan', label='Projected Inflation').set(title='Inflation Over Time', xlabel='Election Year', ylabel='Inflation Rate')
sns.set_theme(style='darkgrid')
plt.show()

#some summary statistics on inflation
print(inflation_results.agg({'US_Inflation':['mean', 'median']}))

#importing new df for these visualizations
df = pd.read_csv('/Users/erikrice/Downloads/Presidential Election Results - Sheet1.csv')
df = df.merge(reference_inflation_df, left_on='Election_Year', right_on='Year', how='outer')
df.drop('Year', axis=1, inplace=True)
df['US_Inflation'] = df['United States of America']
df['US_Inflation'] = df['US_Inflation'].str.replace(' %', '')
df['European_Inflation'] = df['Ø EU']
df['Global_Inflation'] = df['Ø World']      
df.drop(['United States of America','Ø EU', 'Ø World'], axis=1, inplace=True)           
print(df.head())

#isolating inflation results
inflation_results = df[['Election_Year', 'Result', 'Electoral_Votes', 'Vote_Percent', 'Popular_Vote', 'Popular_Vote_Percent', 'US_Inflation', 'Hypothetical']]
inflation_results['US_Inflation'] = inflation_results['US_Inflation'].astype(float)
print(inflation_results.dtypes)
inflation_results['US_Inflation'] = inflation_results['US_Inflation'].replace(' %', '')

#looking at the relationship between inflation and electoral results
sns.scatterplot(data=inflation_results, x='US_Inflation', y='Electoral_Votes', hue='Result').set(title='Relationship Between Inflation and US Electoral Votes', xlabel='Inflation Rate', ylabel='No. Electoral Votes')
plt.show()

#looking at inflation and the popular vote
print(inflation_results.dtypes)
sns.scatterplot(data=inflation_results, x='US_Inflation', y='Popular_Vote', hue='Result').set(title='Relationship Between Inflation and US Popular Vote', xlabel='Inflation Rate', ylabel='No. Votes')
plt.show()

#converting result column to binary
result_binary1 = inflation_results['Result'].astype(str)
result_binary1 = inflation_results['Result'].str.replace('W', '1')
result_binary2 = result_binary1.str.replace('L', '0')

#adding this column to both dataframes
df['Result_Binary'] = result_binary2
inflation_results['Result_Binary'] = result_binary2

#looking at inflation and pure wins/loses
inflation_corr = inflation_results[['US_Inflation', 'Result_Binary']]
print(inflation_corr.corr())

#looking for relationships in the data
inflation_numeric = inflation_results[['Election_Year', 'Electoral_Votes', 'Vote_Percent', 'Popular_Vote', 'Popular_Vote_Percent', 'US_Inflation', 'Result_Binary']]
sns.pairplot(data=inflation_numeric)
plt.show()

#what about a heatmap
sns.heatmap(inflation_numeric.corr(), annot=True)
plt.show()

#pivoting to real GDP and the Yield Curve
#importing this dataset 
yield_curve_df = pd.read_csv('/Users/erikrice/Downloads/cleveland_fed_yieldcurve (1).csv', parse_dates=True)
print(yield_curve_df.head())

#in actual presentation, include source graph when explaining yield curve and recessions 

#cleaning dataset (for eventual merge)
yield_curve_df = yield_curve_df[['DateTime', '10yr_3mo_spread', 'real_gdp_growth']]
yield_curve_df = yield_curve_df.dropna()
yield_curve_df = pd.DataFrame(yield_curve_df)
yield_curve_df['Yield_Curve'] = yield_curve_df['10yr_3mo_spread']
yield_curve_df['Real_GDP'] = yield_curve_df['real_gdp_growth']
yield_curve_df = yield_curve_df[['DateTime', 'Yield_Curve', 'Real_GDP']]
print(yield_curve_df)

#converting and isolating dates (6 months from election is best fit for yield curve)
yield_curve_df['DateTime'] = pd.to_datetime(yield_curve_df['DateTime'])
print(yield_curve_df.dtypes)
yield_curve_df['Year'] = yield_curve_df['DateTime'].dt.year
yield_curve_df['Month'] = yield_curve_df['DateTime'].dt.month
yield_curve_df = yield_curve_df[yield_curve_df['Month'] == 6]
yield_curve_df = yield_curve_df[['Year', 'Yield_Curve', 'Real_GDP']]
print(yield_curve_df)

#merging into main dataframe
df = df.merge(yield_curve_df, how='outer', left_on='Election_Year', right_on='Year')
df.drop('Year', axis=1, inplace=True)

#looking at correlations between these indicators and presidential elections
real_gdp_corr = df[['Real_GDP', 'Result_Binary']]
print(real_gdp_corr.corr())
yield_curve_corr = df[['Yield_Curve', 'Result_Binary']]
print(yield_curve_corr.corr())

#heatmap of all potential correlations with real GDP and the yield curve (very small sample size - numbers don'te tell us much yet)
yield_curve_numeric = df[['Election_Year', 'Electoral_Votes', 'Vote_Percent', 'Popular_Vote', 'Popular_Vote_Percent', 'Result_Binary', 'Yield_Curve', 'Real_GDP']]
sns.heatmap(yield_curve_numeric.corr(), annot=True)
plt.show()

#original reference linegraph with gdp and yield curve
#there have only been five elections where yield curve data is available. isolate these in presentation. 
#limited sample size - only three incumbency elections since this has been measaured - but effect is there. 
#then isolaote the projected gdp growth for next year. does not bode well for Biden. 

#what about the incumbency effect? 
#incumbency column added to source dataframe. reimporting
df = pd.read_csv('/Users/erikrice/Downloads/Presidential Election Results - Sheet1.csv')

#isolating elections where an incumbent was running
incumbency_elections = df[df['Incumbency'] == 'Yes']

#turning into dataframe
incumbency_elections = pd.DataFrame(incumbency_elections)
print(incumbency_elections.head())

#how many of all presidential elections is this?
total_no_presidential_elections = (len(df.index))/2
total_no_incumbency_elections = len(incumbency_elections.index)
prop_incumbency_elections = total_no_incumbency_elections / total_no_presidential_elections
print(prop_incumbency_elections)

#about half of presidential elections have involved an incumbent
Incumbency_Labels = 'Involved Incumbent', 'No Incumbent'
Incumbency_Percent = [45.61, 54.39]
fig1, ax1 = plt.subplots()
ax1.pie(Incumbency_Percent, labels=Incumbency_Labels,autopct='%1.1f%%')
ax1.axis('equal')  
plt.show()

#how often did incumbents win?
winning_incumbents = incumbency_elections[incumbency_elections['Result'] == 'W']
losing_incumbents = incumbency_elections[incumbency_elections['Result'] == 'L']
total_incumbents = winning_incumbents + losing_incumbents
winning_incumbents_per = winning_incumbents / total_incumbents
print(winning_incumbents_per)

#bar graph of absolute results (PROBABLY EXTEND Y MAXIMUM)
data1 = {'Candidate':['Incumbent', 'Not Incumbent'],
        'Elections Won':[17, 9]}
incumbency_to_chart = pd.DataFrame.from_dict(data1)
sns.catplot(x='Candidate', y='Elections Won', data=incumbency_to_chart, kind='bar', ci=None)
plt.show()

#pie chart of percentage
Candidate = 'Incumbent', 'Not Incumbent'
Percent_Elections_Won = [65.38, 34.62]
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
fig1, ax1 = plt.subplots()
ax1.pie(Percent_Elections_Won, labels=Candidate,autopct='%1.1f%%', colors=colors)
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
ax1.axis('equal') 
plt.show() 

#what about just recent American history?
recent_incumbency_elections = incumbency_elections[incumbency_elections['Election_Year'] > 1973]
print(recent_incumbency_elections)

#charting recent incumbency results (PROBABLY EXTEND Y MAXIMUM)
data2 = {'Candidate':['Incumbent', 'Not Incumbent', 'Incumbent', 'Not Incumbent'],
        'Elections Won':[17, 9, 4, 3],
        'Range':['All History', 'All History', 'Last 50 Years', 'Last 50 Years']}
recent_incumbency_to_chart = pd.DataFrame.from_dict(data2)
sns.catplot(x='Range', y='Elections Won', data=recent_incumbency_to_chart, kind='bar', ci=None, hue='Candidate')
plt.show()

#pie chart for recent history
Candidate2 = 'Incumbent', 'Not Incumbent'
Percent_Recent_Elections_Won = [57.14, 42.86]
fig1, ax1 = plt.subplots()
ax1.pie(Percent_Recent_Elections_Won, labels=Candidate2,autopct='%1.1f%%', colors=colors)
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
ax1.axis('equal')  
plt.show()

#pie chart if Biden loses in 2024. inbumbency effect in recent American history is completely erased, and it's not chance.
Candidate3 = 'Incumbent', 'Not Incumbent'
Percent_Recent_Elections_Won_Biden_Loss = [50, 50]
fig1, ax1 = plt.subplots()
ax1.pie(Percent_Recent_Elections_Won_Biden_Loss, labels=Candidate3,autopct='%1.1f%%', colors=colors)
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
ax1.axis('equal')  
plt.show()

#pivoting to our final indicator focus: presidential approval rating
#first term mean, highest and lowest approval rating column added to source dataset. reimporting
df = pd.read_csv('/Users/erikrice/Downloads/Approval Rating Presidential Election Results - Sheet1 (1).csv')

#cleaning dataset
approval_df = df[['Election_Year', 'P_Candidate', 'Result', 'Electoral_Votes', 'Vote_Percent', 'Popular_Vote', 'Popular_Vote_Percent', 'First_Term_Approval', 'Highest_Approval', 'Lowest_Approval']]
approval_df = approval_df.set_index('Election_Year')
print(approval_df.isna())
approval_df = approval_df.dropna()
approval_df = pd.DataFrame(approval_df)
approval_df['First_Term_Approval'] = approval_df['First_Term_Approval'].str.replace('%', '')
approval_df['Highest_Approval'] = approval_df['Highest_Approval'].str.replace('%', '')
approval_df['Lowest_Approval'] = approval_df['Lowest_Approval'].str.replace('%', '')
approval_df[['First_Term_Approval', 'Highest_Approval', 'Lowest_Approval']] = approval_df[['First_Term_Approval', 'Highest_Approval', 'Lowest_Approval']].astype(float)
print(approval_df.dtypes)
print(approval_df.head())

#a cursory look at presidential approval ratings since they began being recorded
sns.catplot(y='P_Candidate', x='First_Term_Approval', data=approval_df, kind='bar', ci=None)
plt.xticks(rotation=90)
plt.show()

#what about their highest recorded polling number?
sns.catplot(x='P_Candidate', y='Highest_Approval', data=approval_df, kind='bar', ci=None)
plt.xticks(rotation=90)
plt.show()

#their lowest?
sns.catplot(x='P_Candidate', y='Lowest_Approval', data=approval_df, kind='bar', ci=None)
plt.xticks(rotation=90)
plt.show()

#graphing these all together. a lot of varience within a term
x1 = approval_df['P_Candidate']
y1 = approval_df['First_Term_Approval']
y2 = approval_df['Highest_Approval']
y3 = approval_df['Lowest_Approval']
approval_chart = pd.DataFrame(np.c_[y1, y2, y3], index=x1)
approval_chart.plot.bar(color=['green', 'blue', 'cyan'])
plt.legend(['Avg. 1st Term', 'Highest', 'Lowest'])
plt.show()

#is there a correlation?
approval_numeric = approval_df[['Electoral_Votes', 'Vote_Percent', 'Popular_Vote', 'Popular_Vote_Percent', 'First_Term_Approval', 'Highest_Approval', 'Lowest_Approval']]
print(approval_numeric.corr())
sns.heatmap(approval_numeric.corr(), annot=True)
plt.show()

sns.catplot(data=df, x='Result', y='P_Age', kind='box')
plt.show()