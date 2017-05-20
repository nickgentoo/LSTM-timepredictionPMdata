import sys
from math import sqrt
file =open(sys.argv[1],"r")
first=True
n=0
sum_RMSE=0.0
sum_MAE=0.0
sum_perc=0.0
for line in file:
#       print line
        if not first:
                L=line.split(",")
                if float(L[5])!=0.0:
                        RMSE=(float(L[5])/(3600*24))**2
                        MAE=float(L[5])/(3600*24)
                        PERC=float(L[5])/float(L[6])
                        #print L[6]
                        #if float(L[6]) < (24*60*60): 
                        sum_RMSE+=RMSE
                        sum_MAE+=MAE
                        sum_perc+=PERC
                        n+=1            
                        #print RMSE
                
        else:
                first=False

total_RMSE=sqrt(sum_RMSE/n)
total_MAE=sum_MAE/n
total_perc=(sum_perc/n)*100.0
print "Results in days:"
print "RMSE:", total_RMSE, "MAE:", total_MAE, "MAPE:",total_perc,"%"
