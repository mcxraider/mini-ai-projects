import pandas as pd 
import os 
from pyrogram import Client 
from dotenv import load_dotenv 
 
# Initialize dataframe to store data after scraping 
columns = ['Group', 'Text', 'Caption'] 
df = pd.DataFrame(columns=columns, dtype=str) 
df2 = pd.DataFrame() 
 
# Get API Keys 
load_dotenv() 
api_id   = os.environ["TELE_API_ID"]
api_hash = os.environ["TELE_API_HASH"] 
 
# Pyrogram config 
app = Client("my_account", api_id, api_hash) 
 
# Get Chat IDs 
chat_ids = [-1001397008848, -1001107383315, -1001210484489] 
 
# Main Code Driver 
async def main(): 
    async with app: 
        for channel in chat_ids: 
            title = await app.get_chat(channel) 
            title = title.title 
            async for message in app.get_chat_history(channel): 
                text = message.text if hasattr(message, 'text') else "" 
                caption = message.caption if hasattr(message, 'caption') else "" 
                df.loc[len(df.index)] = [title, text, caption] 
app.run(main()) 
# Clean Data 
def remove_newlines(s): 
    if isinstance(s, str): 
        return s.replace('\n', '') 
    else: 
        return s 
 
df = df.map(remove_newlines) 
df.to_csv('scrapedmessages.csv', encoding='utf-8', errors='ignore', index=False)