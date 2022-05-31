from __future__ import print_function
import urllib.request
from bs4 import BeautifulSoup
import os

os.chdir("D:/Backup/桌面/pc/")
strYear = '2019'
strFile = 'shenzhenWeather' + strYear + '.csv'
f = open(strFile, 'w')

for month in range(1, 13):
    if (month < 10):
        strMonth = '0' + str(month)
    else:
        strMonth = str(month)
    strYearMonth = strYear + strMonth
    print("\nGetting data for month" + strYearMonth + "...", end='')

    url = "http://lishi.tianqi.com/shenzhen/" + strYearMonth + ".html"
    page = urllib.request.urlopen(url)
    soup = BeautifulSoup(page)
    weatherSet = soup.find(attrs={"class": "thrui"})
    if (weatherSet == None):
        print("fail to get the page", end='')
        continue

    for line in weatherSet.contents:
        if (line.__class__.__name__ == 'NavigableString'): continue
        # if (len(line.attrs) > 0): continue
        contents = line.contents
        if len(contents) < 7:continue
        strDate = contents[1].text[:10]

        a = contents[3].text
        index1 = a.find('℃')
        highWeather = a[:index1]

        b = contents[5].text
        index2 = b.find('℃')
        lowWeather = b[:index2]

        c = contents[7].text
        if '雨' in c:
            weathertype = str(1)
        else:
            weathertype = str(0)
        f.write(strDate + ',' + lowWeather + ',' + highWeather + ',' + weathertype + '\n')
    print("done", end='')

f.close()
print("\nover")