# Contact-Mutation-Regression

Setup
=====
1. <code>git clone https://github.com/Nightknight3000/Contact-Mutation-Regression</code>
2. <code>pip install -r requirements.txt</code>

Usage
=====
## The CLI - Command Line Interface
```bash
> python start_mhc_1_prediction_tool.py
usage: start_contact_mutation_regression.py -c <learning algorithm> -i <inputfilepath> -p <inputfilepath> -o <outputpath> [-s/-v]
-c <arg> either xgb or sklearn to specify learning algorithm
-i <arg> csv/txt-file for classifier training
-p <arg> csv/txt-file to be predicted by classifier (optional)
-o <arg> output path for classified inputfile (optional)
-s/-v, --silent/--verbose provide -v or --verbose to see full classification process and training (default -s)
(currently only works with xgb)
```

Examples
=====
Using the example file 'contact_map_blomap_6A.csv' as first inputfile in the data folder:
[data](https://github.com/Nightknight3000/Contact-Mutation-Regression/tree/master/data/inputfiles).

Now we can run the project using our shell of choice:

<code>python start_contact_mutation_regression.py -c xgb -i data/inputfiles/contact_map_blomap_6A.csv -p False -o False</code>

Authors
=====
Team iGEM 2018 Tübingen <br />
Lukas Heumos <br />
Steffen Lemke <br />
Alexander Röhl