import pandas as pd


def get_top10(datas, year, column):
    year_data = datas[datas['year'] == year]
    top10_data = year_data.sort_values(by=column, ascending=False)[:10]
    return top10_data[['batter_name', column]]


def print_top10(datas, year):
    columns = ['H', 'avg', 'HR', 'OBP']
    top10 = {}

    for column in columns:
        top_data = get_top10(datas, year, column)
        top10[f'{column}_player'] = top_data['batter_name'].values
        top10[column] = top_data[column].values

    top_df = pd.DataFrame(top10, index=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    top_df.name = f'(1) top 10 in {year}'
    top_df.index.name = 'rank'

    print(top_df.name + ' is')
    print(top_df)
    print('\n')


def get_highest_wars(datas):
    war_position = []
    war_player = []
    war = []

    for x in positions:
        position = datas[datas['cp'] == x]
        position = position.sort_values(by='war', ascending=False)[:1]
        war_position.append(position['cp'].values)
        war_player.append(position['batter_name'].values)
        war.append(position['war'].values)

    return war_position, war_player, war


data = pd.read_csv('2019_kbo_for_kaggle_v2.csv')

print_top10(data, 2015)
print_top10(data, 2016)
print_top10(data, 2017)
print_top10(data, 2018)

positions = ['포수', '1루수', '2루수', '3루수', '유격수', '좌익수', '중견수', '우익수']
data2018 = data[data['year'] == 2018]
highest_wars_position, highest_wars_player, highest_wars = get_highest_wars(data2018)

war_info = {'positions': highest_wars_position,
            'player': highest_wars_player,
            'war': highest_wars}
war_data = pd.DataFrame(war_info)

print('(2) The player with the highest war by position in 2018 is ')
print(war_data)
print()

highest_corr = 0
highest_corr_with = ''
corr_with = ['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG']

for x in corr_with:
    corr = data['salary'].corr(data[x])
    if highest_corr < corr:
        highest_corr = corr
        highest_corr_with = x


print("(3) " + highest_corr_with + " has the highest correlation with salary")
print("correlation between salary and " + highest_corr_with + " is " + str(highest_corr))
