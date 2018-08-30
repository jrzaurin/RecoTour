import pandas as pd

def dump_libffm_file(df, target, catdict, current_code, cat_codes, f, verbose=False):
    noofrows = df.shape[0]
    noofcolumns = len(df.columns)
    with open(f, "w") as libffm_file:
        for n, r in enumerate(range(noofrows)):
            if verbose:
                if((n%100000==0) and n!=0): print('Row',n)
            datastring = ""
            datarow = df.iloc[r].to_dict()
            datastring += str(datarow[target])
            for i, x in enumerate(catdict.keys()):
                if(catdict[x]==0):
                    datastring = datastring + " "+str(i)+":"+ str(i)+":"+ str(datarow[x])
                else:
                    if(x not in cat_codes):
                        cat_codes[x] = {}
                        current_code +=1
                        cat_codes[x][datarow[x]] = current_code
                    elif(datarow[x] not in cat_codes[x]):
                        current_code +=1
                        cat_codes[x][datarow[x]] = current_code
                    code = cat_codes[x][datarow[x]]
                    datastring = datastring + " "+str(i)+":"+ str(int(code))+":1"
            datastring += '\n'
            libffm_file.write(datastring)
    return current_code, cat_codes
