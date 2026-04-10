import pandas as pd

def check_busyness():
    df = pd.read_csv('/Users/chris/Desktop/Yelp JSON/yelp_dataset/restaurant_venue_features.csv')
    nz = (df['peak_busyness'] > 0).sum()
    total = len(df)
    maxx = df['peak_busyness'].max()
    print(f"Foursquare peak_busyness: {nz} out of {total} > 0. Max value: {maxx}")
    
if __name__ == "__main__":
    check_busyness()
