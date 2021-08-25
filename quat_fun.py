from pyquaternion import Quaternion


q1 = Quaternion(axis=[1,0,1], angle = 3.141592).normalised
q2 = Quaternion(axis=[0,5,0], angle = 3.141592/2).normalised

q_d = q1.conjugate * q2
q_d =q_d.normalised
print('q1: {}'.format(q1))
print('q2: {}'.format(q2))

q2_recon = q1 * q_d
print('q2_recon: {}'.format(q2_recon))
