import pymysql as MySQLdb
import pandas as pd
from tqdm import tqdm

from warnings import filterwarnings
filterwarnings('ignore', category = MySQLdb.Warning)


source_database="reddit"
your_database="self_beliefs"
your_table=f"{your_database}.askreddit_comments_15_19"
source_subreddits = ["AskReddit"] 
year_month = pd.date_range('2015-01-01','2019-12-31', freq='MS').strftime("%Y_%m").tolist()
print("Year Months to Read",year_month)

# Create the table if not exists
db = MySQLdb.connect(db=your_database, read_default_file='~/.my.cnf', use_unicode=True, charset="utf8mb4")
cur = db.cursor()
cur.execute(f"""CREATE TABLE IF NOT EXISTS {your_table} (message_id varchar(70) primary key, user_id varchar(128), message text, subreddit varchar(64))""")
cur.execute(f"""CREATE INDEX IF NOT EXISTS idx_message_id ON {your_table} (message_id)""")
db.commit()
db.close()

for subreddit in source_subreddits:
    db = MySQLdb.connect(db=source_database, read_default_file='~/.my.cnf', use_unicode=True, charset="utf8mb4")
    cur = db.cursor()
    
    for ym in tqdm(year_month):
        
        # Get Comments
        try:
            cur.execute(f"""
                        INSERT IGNORE INTO {your_table} (
                            SELECT message_id, user_id, message, subreddit 
                            FROM {source_database}.com_{ym} 
                            WHERE subreddit = "{subreddit}" 
                                AND message != "[removed]" 
                                AND message != "[deleted]" 
                                AND message LIKE "%I am % person%"
                            ORDER BY RAND()
                        )""" )
        except Exception as e:
            print(f"FAILED for {ym}, {e} -- comments")
            
        # Get Posts
        # try:
        #     cur.execute(f"""
        #                 INSERT IGNORE INTO {your_table} (
        #                     SELECT message_id, user_id, message, subreddit 
        #                     FROM {source_database}.sub_{ym} 
        #                     WHERE subreddit="{subreddit}" 
        #                         AND message!="[removed]" 
        #                         AND message != "[deleted]" 
        #                         AND (message LIKE "%I %" OR message LIKE "%I\'m%")
        #                     ORDER BY RAND()
        #                 )""" )
        # except Exception as e:
        #     print(f"FAILED for {ym}, {e} -- subreddit posts")
            
    db.commit()
    db.close()