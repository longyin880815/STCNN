
class Path(object):
    @staticmethod
    def db_root_dir():
        return '/home/xk/Dataset/DAVIS'

    @staticmethod
    def save_root_dir():
        return './output'

    @staticmethod
    def models_dir():
        return "./models"

    @staticmethod
    def data_dir():
        return "./data"

    @staticmethod
    def VID_list_file():
        return "./data/VID_seqs_list.txt"

    @staticmethod
    def DAVIS_list_file():
        return "./data/DAVIS_seqs_list.txt"

    @staticmethod
    def MSRAdataset_dir():
        return "/home/xk/Dataset/MSRA10K_Imgs_GT/MSRA10K_Imgs_GT/Imgs/"

    @staticmethod
    def VOC_dir():
        return "/home/xk/Dataset/VOC/"