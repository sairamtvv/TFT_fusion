# TFT_fusion
What were the problems that we were facing

To understand this let us remember all the steps
1) read data from ur excel for all the sites ---*** NO PROBLEM
2) you want to set frequency to 30 seconds --- *** No PROBLEM
3) we have different types of NANS 
	I) We got NANS during night -- In the start we will not remove them, but we will fill all of the night with zero so that there are no missing values 
									--- *** No PROBLEM
	II) There are nans for continous of 10mins (less than 20 data points), these can be forward or backward filled --- *** No PROBLEM

	III) If is there nan values for a  day or two in misssle (this can be imputed).	
		
	IV) What to do if we have nans for continously for number of days
		
	V) Adding weights to the time stamps - where we have nans for continuous. 

	VI) If you are running a model that is not a time series but a regression model, (Then it does not matter) 	
	
	VII) There are static variables (this effect may not be important for you)
	
		
	
2) Only after filling the nans we can convert to the actual types, PROBLEM we were facing were:
	1) First it was showing no nan values , but after converting its type , it is showing that there are many NAN values -- YET to be DONE


How did we remove the repeated time stamp values.@gurpreet

3) Methodology for Missing Values:
   1) set freq as day
	2) Python will question: What should I do with the data that is at half 
   an minute interval
3) 
4) 


Figure out how many days are missing ---- information

1. Missing dates-- taken care after 30s freq sampling
2. missing empty strings-- filled with str to nan conversion
3. missing NAN str 
4. First put the values as 0 for night condition and IscRef <= 0.6 
5. Take nan(counts) as a function of date. Decide the threshold 
and mark the dates where nan are large. They will be given less 
weights. 
   Check how many nans in sequence
   if its for few mins or hrs: fill with the average -- give least weightage
   if for whole day: fill the whole day with the previous day -- give least weightage 
   For 10 days, take the mean of the whole month for that particular time.

hour_limit:  

-----------------------------------------------------------------
12/3 0.25 filled 
-----
convert 30s to hr basis data and then look for NAN in more than 1 day
how ffill work limit=120

