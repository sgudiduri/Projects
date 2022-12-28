import urllib.request as urllib2
import re
import ast
import pandas as pd

def ReadPage(url):
    try:
        openedPage = urllib2.urlopen(url)
    except:
        print("Could not open %s" % url)
        return
    pageContents = str(openedPage.read())
    listOfArrays = re.findall(r'\{"@type":"ListItem[A-Za-z0-9:\",\s/#.-]*}', pageContents)
    outputArray = {}
    for array in listOfArrays:

        tempdict= ast.literal_eval(array)

        """        if "/" in tempdict["name"]:
            splitNames= tempdict["name"].split(" / ")
            outputArray[splitNames[0]] = tempdict["position"]
            outputArray[splitNames[1]] = tempdict["position"]
        else:"""
        outputArray[tempdict["name"]] = tempdict["position"]

    return outputArray

if __name__ == "__main__":
    bestSchool = ReadPage("https://www.neighborhoodscout.com/ia/ames/schools")
    safestForCrime = ReadPage("https://www.neighborhoodscout.com/ia/ames/crime")
    largestIncomeGrowth = ReadPage("https://www.neighborhoodscout.com/ia/ames/demographics")
    largestHomeAppreciationSince2000 = ReadPage("https://www.neighborhoodscout.com/ia/ames/real-estate")
    mostExpensive = ReadPage("https://www.neighborhoodscout.com/ia/ames")
    neighborhoods = set(bestSchool.keys())

    neighborhoods = neighborhoods.union(set(safestForCrime.keys()),
                        set(largestIncomeGrowth.keys()),
                        set(largestHomeAppreciationSince2000.keys()),
                        set(mostExpensive.keys()))

    df = pd.DataFrame([], index=neighborhoods, columns=[])
    dftemp = pd.DataFrame.from_dict(bestSchool, orient='index', columns=['BestSchool'])
    df = pd.merge(df, dftemp, left_index=True, right_index=True, how="left")
    dftemp = pd.DataFrame.from_dict(safestForCrime, orient='index', columns=['Safest'])
    df = pd.merge(df, dftemp, left_index=True, right_index=True, how="left")
    dftemp = pd.DataFrame.from_dict(largestIncomeGrowth, orient='index', columns=['LargestIncomeGrowth'])
    df = pd.merge(df, dftemp, left_index=True, right_index=True, how="left")
    dftemp = pd.DataFrame.from_dict(largestHomeAppreciationSince2000, orient='index', columns=['LargestHomeAppreciationSince2000'])
    df = pd.merge(df, dftemp, left_index=True, right_index=True, how="left")
    dftemp = pd.DataFrame.from_dict(mostExpensive, orient='index', columns=['MostExpensive'])
    df = pd.merge(df, dftemp, left_index=True, right_index=True, how="left")
    df.fillna(value=11, inplace=True)
    df.to_csv("Ames Neighborhood Rankings.csv")