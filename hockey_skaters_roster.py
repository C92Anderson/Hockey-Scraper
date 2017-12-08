import requests
import pandas as pd
import math
import pymysql as pg 
import pymysql as pg
import datetime

skatersAll = pd.DataFrame()

seasons = ["20032004","20052006","20062007","20072008","20082009","20092010","20102011","20112012","20122013","20132014","20142015","20152016","20162017","20172018"]

for season in seasons:

    url = "http://www.nhl.com/stats/rest/grouped/skaters/basic/season/bios?cayenneExp=seasonId=" + str(season)

    # Read skaters
    skaters = requests.get(url).json()

    skaters = pd.DataFrame(skaters['data'])

    skatersAll = skatersAll.append(skaters)

skatersAll['cron_ts'] = datetime.datetime.now()

skatersAll = skatersAll[~skatersAll['playerId'].isin([8474744,8466208,8471747,8468436,8466155,8476979,8471221])]

###insert into phpmyadmin
conn2 = pg.connect(host='mysql.crowdscoutsports.com', user='ca_elo_games', password='cprice31!', port=3306, db='nhl_all')

skatersAll.to_sql(con=conn2, name='hockey_skaters_roster', if_exists='replace', flavor='mysql')

conn.commit()
conn.close()