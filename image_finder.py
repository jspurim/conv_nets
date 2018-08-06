import argparse
from googleapiclient.discovery import build
import pprint

parser = argparse.ArgumentParser(prog = "image_find", description="Uses google custom search API to search for images matching the provided query.")
parser.add_argument("--key", required=True, help="API key.")
parser.add_argument("--gcse", required=True, help="Google custom search engine id.")
parser.add_argument("-n", type=int, help="Number of images to fetch.")
parser.add_argument("query", metavar="TERM", nargs="+", help="Terms to search for.")

args = parser.parse_args()

service = build("customsearch", "v1", developerKey=args.key)

def getImages(query, s):
    resp = service.cse().list(q=query, cx=args.gcse, num=10, start=s, searchType="image", imgType="photo", imgColorType="color").execute()
    items = resp.items()
    results = None
    for item in items:
        if item[0] == "items":
            results = item[1]
            break
    return [result["link"] for result in results]

n = args.n
query = " ".join(args.query)

imgLinks = []

s = 1
while len(imgLinks) < n:
    results = getImages(query, s)
    s += 10
    imgLinks = imgLinks + results

for link in imgLinks:
    print link
