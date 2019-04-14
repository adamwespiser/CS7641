## TO Install
$ git clone https://github.com/adamwespiser/CS7641.git    
$ cd CS7641/assignment4    
$ git pull origin gce    
$ virtualenv venv    
$ source venv/bin/activate    
$ pip install -r requirements.txt    

## To Run experiments and do half the plotting    
Note this will take a while, {'Q': ~6 hours, 'PI': 80 mins, 'VI': 14 mins}    
$ ./run_mcp.sh    

## to finish the plotting    
Make sure you have R, and ggplot2 installed    
Open up an R interpreter in CS7641/assignment4 and run    
> source('q_learning_plots.R')    
