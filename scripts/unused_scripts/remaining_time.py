import time
import sys
import matplotlib.pyplot as plt


def remaining_time(start, job_id, total_jobs):
	run_time = time.time() - start_time
	return (run_time/job_id)*(total_jobs-job_id)



def readablesecs(time_in_secs):
	hours = int(time_in_secs/3600)
	minutes = int((time_in_secs - (hours*3600))/60)
	secs = float(time_in_secs - (hours*3600) - (minutes*60))

	hours_str =  ('0' + str(hours))[-2:]
	minutes_str = ('0' + str(minutes))[-2:]
	secs_str = ('0' + str(secs).split('.')[0])[-2:] + '.' + (str(secs).split('.')[1] + '00')[:2]

	return '%s:%s:%s' %(hours_str, minutes_str, secs_str)






my_stuff = [x for x in range(1000000)]


counter = 0
predicted_times = []
start_time = time.time()
for thing in my_stuff:
	y = thing * counter + 10000 + counter
	if counter >= 1:
		# sys.stdout.write('\r' + readablesecs(remaining_time(start_time, counter, len(my_stuff))))
		# sys.stdout.flush()
		predicted_times.append(remaining_time(start_time, counter, len(my_stuff)))
	counter += 1


fig, ax = plt.subplots()
ax.plot([x for x in range(1, len(predicted_times)+1)], predicted_times)
fig.savefig('time_test.png')
plt.close()
