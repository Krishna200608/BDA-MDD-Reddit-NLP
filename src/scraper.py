import requests
import pandas as pd
import time
from datetime import datetime
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PullPushScraper:
    def __init__(self):
        self.base_url = "https://api.pullpush.io/reddit/search/submission"
        
    def fetch_posts(self, subreddit, limit=5000):
        url = self.base_url
        posts = []
        before = int(time.time())
        retries = 3
        
        logging.info(f"Targeting {limit} posts from r/{subreddit}")
        
        with tqdm(total=limit, desc=f"r/{subreddit}") as pbar:
            while len(posts) < limit:
                params = {
                    "subreddit": subreddit,
                    "size": 100, # PullPush maximum size limit per request is 100
                    "before": before,
                    "sort": "desc"
                }
                
                try:
                    response = requests.get(url, params=params, timeout=10)
                    response.raise_for_status()
                    data = response.json().get('data', [])
                    
                    if not data:
                        logging.warning(f"No more data returned for r/{subreddit}. Reached epoch: {before}")
                        break
                    
                    for item in data:
                        # Extract the needed fields
                        post_dict = {
                            "post_id": item.get('id'),
                            "subreddit": item.get('subreddit'),
                            "timestamp": datetime.fromtimestamp(item.get('created_utc', 0)).isoformat(),
                            "title": item.get('title', ''),
                            "selftext": item.get('selftext', ''),
                            "score": item.get('score', 0),
                            "num_comments": item.get('num_comments', 0),
                            "author": item.get('author', ''),
                        }
                        # We specifically want selftext that has actual content
                        if post_dict['selftext'] not in ['[removed]', '[deleted]', '']:
                            posts.append(post_dict)
                            pbar.update(1)
                            
                        # Keep track of the oldest timestamp for pagination
                        last_utc = item.get('created_utc')
                        if last_utc and last_utc < before:
                            before = last_utc
                            
                    time.sleep(1.0) # Rate limiting to be polite
                    
                except requests.exceptions.RequestException as e:
                    logging.error(f"Error fetching data: {e}. Retrying... ({retries} left)")
                    retries -= 1
                    time.sleep(3)
                    if retries == 0:
                        break

        # Trim exact amount
        posts = posts[:limit]
        logging.info(f"Finished extracting {len(posts)} posts from r/{subreddit}")
        return posts

if __name__ == "__main__":
    # Test execution
    scraper = PullPushScraper()
    df = scraper.fetch_posts("CasualConversation", limit=100)
    print(f"Test Successful. Extracted {len(df)} sample posts.")
