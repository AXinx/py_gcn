
from util.utils import read_file
import matplotlib.pyplot as plt

#load data
epoch = read_file('epoch','txt')
avg_loss = read_file('avg_loss', 'txt')
avg_acc = read_file('avg_acc', 'txt')
val_loss = read_file('val_loss', 'txt')

#plot
s = 10
fig = plt.figure()
plt.title('Loss and accuracy', fontsize = s)
ax1 = fig.add_subplot(1,1,1)
l1 = ax1.plot(epoch, avg_loss, 'r', label='train loss')
l3 = ax1.plot(epoch, val_loss, 'y', label='val loss')
#plt.legend(bbox_to_anchor=(1.0,0.15))
ax1.set_ylabel('Loss',fontsize = s)
ax2 = ax1.twinx()
l2 = ax2.plot(epoch, avg_acc, 'g', label='train accuracy')
ls = l1+l2+l3
labs = [l.get_label() for l in ls]
ax1.legend(ls, labs, bbox_to_anchor=(1.0,0.95))
ax2.set_ylabel('Accuracy',fontsize = s)
ax1.set_xlabel('Epoch',fontsize = s)
plt.show()
plt.savefig('loss_acu.png')


