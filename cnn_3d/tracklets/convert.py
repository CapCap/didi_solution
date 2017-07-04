from __future__ import print_function

from velodyne_msgs import msg as v_msg
import rosbag
import rospy

bag = rosbag.Bag('/Users/max/Desktop/carchallenge/Didi-Release-2/data/1/10.bag', 'r')

topics = bag.get_type_and_topic_info()[1].keys()
print("\n".join(topics))

for topic, msg, t in bag.read_messages(topics=['/velodyne_packets']):
    import pdb
    pdb.set_trace()
    header=msg.header, packets=msg.packets
    #print(topic, t, msg)
    system.exit()


 rospy.init_node('listener', anonymous=True)
rospy.spin()