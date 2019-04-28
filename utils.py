import pickle

# CONSTANT
categories = ['duration', 
    'protocol_type', 
    'service',
    'flag',
    'src_bytes',
    'dst_bytes',
    'land',
    'wrong_fragment',
    'urgent',
    'hot',
    'num_failed_logins',
    'logged_in',
    'num_compromised',
    'root_shell',
    'su_attempted',
    'num_root',
    'num_file_creations',
    'num_shells',
    'num_access_files',
    'num_outbound_cmds',
    'is_host_login',
    'is_guest_login',
    'count',
    'srv_count',
    'serror_rate',
    'srv_serror_rate',
    'rerror_rate',
    'srv_rerror_rate',
    'same_srv_rate',
    'diff_srv_rate',
    'srv_diff_host_rate',
    'dst_host_count',
    'dst_host_srv_count',
    'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate',
    'dst_host_srv_serror_rate',
    'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate',
    'label']


# class Engine():
#     def __init__(self, model_file):
#         self.model = pickle.load(open(model_file, 'rb'))
#     def predict(self, X_test):
#         y_pred = self.model.predict(X_test)
#         return y_pred


def load_model(model_file):
    return pickle.load(open(model_file, 'rb'))

def custom_predict(model_file, X_test):
    model = load_model(model_file)
    return model.predict(X_test)
