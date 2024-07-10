import cv2
import numpy as np
from mseg_semantic.tool.inference_task import InferenceTask
import argparse
from mseg_semantic.utils.config import CfgNode
from mseg_semantic.utils import config
from pathlib import Path
import mseg.utils.names_utils as names_utils
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class SemSegNode(Node):

    def __init__(self):
        super().__init__('sem_seg_node')
        self.publisher_ = self.create_publisher(Image, '/camera/semseg/image', 10)
        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.listener_callback,
            1)
        self.subscription  # prevent unused variable warning

        self.package_dir = str(Path(__file__).resolve().parents[1])

        args = self.get_parser()

        assert isinstance(args.model_name, str)
        assert isinstance(args.model_path, str)

        if args.dataset == "default":
            args.dataset = Path(args.input_file).stem

        if "scannet" in args.dataset:
            args.img_name_unique = False
        else:
            args.img_name_unique = True

        args.u_classes = names_utils.get_universal_class_names()

        args.print_freq = 10

        args.split = "test"
        args.num_model_classes = len(args.u_classes)

        print(args)

        self.itask = InferenceTask(
            args,
            base_size=args.base_size,
            crop_h=args.test_h,
            crop_w=args.test_w,
            input_file=args.input_file,
            model_taxonomy="universal",
            eval_taxonomy="universal",
            scales=args.scales,
        )

        self.itask.model.eval()

        self.get_logger().info('Semantic Segmentation Node has been started')


    def listener_callback(self, msg):
        rgb_img = np.array(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, -1))
        pred_label_img = self.itask.execute_on_img(rgb_img)

        # Convert pred_label_img to sensor_msgs Image message
        pred_img = Image()
        pred_img.header = msg.header  # Set the header of the message
        pred_img.height = pred_label_img.shape[0]  # Set the height of the image
        pred_img.width = pred_label_img.shape[1]  # Set the width of the image
        pred_img.encoding = "mono8"  # Set the encoding of the image to RGB
        pred_img.is_bigendian = False  # Set the endianness of the image
        pred_img.step = pred_label_img.shape[1]  # Set the step size of the image
        pred_img.data = pred_label_img.flatten().tolist()  # Flatten the image and convert it to a list

        self.publisher_.publish(pred_img)
        

    def get_parser(self) -> CfgNode:
        """ """
        parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation")
        parser.add_argument(
            "--config", type=str, default= (self.package_dir + "/cfg/default_config_360_ms.yaml"), help="config file"
        )
        # parser.add_argument(
        #     "--file_save", type=str, default="default", help="eval result to save, when lightweight option is on"
        # )
        parser.add_argument(
            "opts", help="see config/ade20k/ade20k_pspnet50.yaml for all options", default=None, nargs=argparse.REMAINDER
        )  # model path is passed in
        args = parser.parse_args()
        assert args.config is not None
        cfg = config.load_cfg_from_cfg_file(args.config)
        if args.opts is not None:
            cfg = config.merge_cfg_from_list(cfg, args.opts)
        return cfg

    def imread_rgb(self, img_fpath: str) -> np.ndarray:
        """
        Returns:
            RGB 3 channel nd-array with shape (H, W, 3)
        """
        bgr_img = cv2.imread(img_fpath, cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        rgb_img = np.float32(rgb_img)
        return rgb_img


def main(args=None):
    rclpy.init(args=args)

    sem_seg_node = SemSegNode()

    rclpy.spin(sem_seg_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    sem_seg_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

