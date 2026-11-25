# import functools

# 显示卷积结果
# import matplotlib.pyplot
import numpy
# save
import skimage
from skimage import io
import pathlib
import os
import puremagic
import colour




# from src.lib_convolution_core.attention_weight import (
#     central_peripheral_weight__norm__normal_distribution,
#     generate_distribution_kernel
# )
#
# from src.lib_convolution_core.contrast_gradient import (
#     central_peripheral_contrast__delta_e_cie2000_lch,
#     central_peripheral_contrast__oklch,
#     central_peripheral_contrast__dynamic_kernel
# )


# from src.lib_threshold.threshold import (
#     calculate_sigmoid_slope_from_upper_lower_bound,
#     expand_difference_by_sigmoid_threshold
# )

from src.lib_human_eye_like import (
    central_peripheral_contrast_convolution_local_normalized
)

def normalize_image(image: numpy.ndarray, output_range: tuple = (0, 255)) -> numpy.ndarray:
    """
    将图像归一化到指定的输出范围。

    :param image: 输入图像，类型为 numpy.ndarray
    :param output_range: 输出范围，默认为 (0, 255)
    :return: 归一化后的图像，类型为 numpy.ndarray
    """
    # 获取输入图像的最小值和最大值
    min_val = numpy.min(image)
    max_val = numpy.max(image)

    # 将图像归一化到 [0, 1] 范围
    normalized_image = (image - min_val) / (max_val - min_val)

    # 将归一化后的图像缩放到指定的输出范围
    output_min, output_max = output_range
    scaled_image = normalized_image * (output_max - output_min) + output_min

    return scaled_image

def is_image_file(file_name: str) -> bool:
    """
    检查文件名是否是图像文件。

    :param file_name: 文件名
    :return: 如果是图像文件，返回 True；否则返回 False
    """
    # puremagic
    file_mime_type: list[puremagic.PureMagicWithConfidence] = puremagic.magic_file(file_name)

    return any(mime.mime_type.startswith("image/") for mime in file_mime_type)




def process_image(
    original_image_path: str,
    # canny_image_folder_path: str,
    new_algorithm_folder_path: str,
    # sigmoid_threshold: float = 12,
    # sigmoid_slope_upper_bound: float = 0.95,
    # sigmoid_slope_lower_bound: float = 0.05,
    # sigmoid_slope_upper_value_x: float = 20,
    # sigmoid_slope_lower_value_x: float = 10,
    kernel_size: int = 3,   # 单核心，不叠加
    image_pixel_output_range: tuple = (0, 255),
) -> tuple[str, str]:
    """
    处理图像，应用中心-周边对比卷积，并保存结果。
    该函数将读取原始图像，应用卷积操作，然后进行阈值处理，并保存卷积结果和Canny边缘检测结果。
    该函数返回一个包含原始图像路径、卷积结果保存路径、阈值处理后的卷积结果保存路径以及Canny边缘检测结果保存路径的元组。

    :param original_image_path: 原始图像的路径
    # :param canny_image_folder_path: Canny边缘检测结果保存的文件夹路径
    :param new_algorithm_folder_path: 新算法结果保存的文件夹路径
    # :param sigmoid_threshold: Sigmoid阈值
    # :param sigmoid_slope_upper_bound: Sigmoid斜率上界
    # :param sigmoid_slope_lower_bound: Sigmoid斜率下界
    # :param sigmoid_slope_upper_value_x: Sigmoid斜率上界对应的 x 值
    # :param sigmoid_slope_lower_value_x: Sigmoid斜率下界对应的 x 值
    :param kernel_size: 卷积核的大小
    :param image_pixel_output_range: 图像像素的输出范围
    :return: 包含原始图像路径、卷积结果保存路径、阈值处理后的卷积结果保存路径以及Canny边缘检测结果保存路径的元组
    """
    # =========================== Init ==========================
    # Get name of the original image file
    original_image_name = pathlib.Path(original_image_path).stem  # 获取文件名（不带扩展名）
    # =========================== Read the original image ===========================
    image = io.imread(original_image_path)
    # image_lab = color.rgb2lab(image)  # 转换为 Lab 颜色空间

    # # 计算颜色平均值
    # image_lab_mean = numpy.mean(image_lab, axis=(0, 1))
    # # 归一化颜色平均值
    # image_lab_mean = image_lab_mean / 255.0  # 将平均值归一化到 [0, 1] 范围

    # =========================== New Algorithm: Central-Peripheral Contrast Convolution ===========================
    oklab_image: numpy.ndarray = colour.convert(image, "Output-Referred RGB", "Oklab")
    oklab_image = skimage.color.lab2lch(oklab_image)  # 转换为 LCH 颜色空间
    # L: [0, 1]
    # C: [0, 1]
    # H: [0, 2pi]
    convolution_result = central_peripheral_contrast_convolution_local_normalized(
        pixel_matrix=oklab_image,
        kernel_size=kernel_size,
    )

    # -------------------------- Save the convolution result --------------------------
    # 转为 uint8 类型
    convolution_result_save: numpy.ndarray
    # convolution_result_save = convolution_result.astype(numpy.uint8)
    #
    # convolution_result_save_path: str = (f"{new_algorithm_folder_path}/"
    #                                      f"non_threshold/"
    #                                      f"{original_image_name}_k_{kernel_size}.png")
    # skimage.io.imsave(
    #     convolution_result_save_path,
    #     convolution_result_save
    # )
    # =========================== Normalization ===========================
    # 将卷积结果归一化到指定的输出范围
    convolution_result = normalize_image(
        image=convolution_result,
        output_range=image_pixel_output_range
    )

    # --------------------- Save the normalized convolution result ----------------
    # 转为 uint8 类型
    convolution_result_save = convolution_result.astype(numpy.uint8)
    convolution_result_save_path_normalized: str = (f"{new_algorithm_folder_path}/"
                                                    # f"normalized/"
                                                    f"{original_image_name}_k_{kernel_size}.png")
    skimage.io.imsave(
        convolution_result_save_path_normalized,
        convolution_result_save
    )


    #
    # # =========================== Sigmoid Thresholding ===========================
    #
    # slope: float = calculate_sigmoid_slope_from_upper_lower_bound(
    #     upper_bound=sigmoid_slope_upper_bound,
    #     lower_bound=sigmoid_slope_lower_bound,
    #     upper_value_x=sigmoid_slope_upper_value_x,
    #     lower_value_x=sigmoid_slope_lower_value_x
    # )
    #
    # # 阈值截断
    # convolution_result = expand_difference_by_sigmoid_threshold(
    #     result=convolution_result,
    #     threshold=sigmoid_threshold,
    #     slope=slope,  # --math.log(0.1 / 1.9),
    #     result_range=image_pixel_output_range
    # )
    #
    # # -------------------------- Save the thresholded convolution result --------------------------
    # # 转为 uint8 类型
    # convolution_result_save = convolution_result.astype(numpy.uint8)
    # convolution_result_save_path_threshold = (f"{new_algorithm_folder_path}/"
    #                                          f"threshold/"
    #                                          f"{original_image_name}_k_{kernel_size}.png")
    # skimage.io.imsave(
    #     convolution_result_save_path_threshold,
    #     convolution_result_save
    # )
    #

    # # =========================== Canny Edge Detection ===========================
    # # Apply Canny edge detection
    # # 采用默认值
    # # sigmoid 与自定义算法保持一致 == 1
    # # 因为原先的算法中，不撇除任何信息，没有canny强制归零的high threshold，low threshold.
    # # 所以继续采取默认值，让canny保持参数上的优势
    # canny_edges: numpy.ndarray = skimage.feature.canny(
    #     image=convolution_result,
    # )
    #
    # # -------------------------- Save the Canny edge detection result --------------------------
    # # 转为 uint8 类型
    # canny_edges_save: numpy.ndarray = (canny_edges * 255).astype(numpy.uint8)
    #
    # canny_image_save_path: str = (f"{canny_image_folder_path}/"
    #                   f"{original_image_name}_k_{kernel_size}.png")
    # skimage.io.imsave(
    #     canny_image_save_path,
    #     canny_edges_save
    # )

    return (
        original_image_path,
        convolution_result_save_path_normalized,
        # canny_image_save_path
    )

def process_images_in_folder(
    image_root_path: str,
    # sigmoid_threshold: float = 12,
    # sigmoid_slope_upper_bound: float = 0.95,
    # sigmoid_slope_lower_bound: float = 0.05,
    # sigmoid_slope_upper_value_x: float = 20,
    # sigmoid_slope_lower_value_x: float = 10,
    kernel_size: int = 3,   # 单核心，不叠加
    image_pixel_output_range: tuple = (0, 255),
) -> list[tuple[str, str]]:
    """
    批量处理文件夹中的图像，应用中心-周边对比卷积，并保存结果。

    :param image_root_path: 图像文件夹的根路径
    # :param sigmoid_threshold: Sigmoid阈值
    # :param sigmoid_slope_upper_bound: Sigmoid斜率上界
    # :param sigmoid_slope_lower_bound: Sigmoid斜率下界
    # :param sigmoid_slope_upper_value_x: Sigmoid斜率上界对应的 x 值
    # :param sigmoid_slope_lower_value_x: Sigmoid斜率下界对应的 x 值
    :param kernel_size: 卷积核的大小
    :param image_pixel_output_range: 图像像素的输出范围
    :return: 包含原始图像路径、卷积结果保存路径的元组列表
    """

    # =========================== Init ==========================
    # 创建保存卷积结果的文件夹
    """
    ./base (不需要创建，这个是原图文件夹）
    ./canny
    ./new_algorithm
        ./non_threshold
        ./threshold
    """
    original_image_folder_path: str = os.path.join(image_root_path, "base")
    # canny_image_folder_path: str = os.path.join(image_root_path, "canny")
    new_algorithm_folder_path: str = os.path.join(image_root_path, "new_algorithm")
    # new_algorithm_convolution_folder_path: str = os.path.join(new_algorithm_folder_path, "non_threshold")
    # new_algorithm_normalized_folder_path: str = os.path.join(new_algorithm_folder_path, "normalized")
    # new_algorithm_threshold_folder_path: str = os.path.join(new_algorithm_folder_path, "threshold")

    # 创建文件夹（如果不存在）
    # os.makedirs(canny_image_folder_path, exist_ok=True)
    # os.makedirs(new_algorithm_convolution_folder_path, exist_ok=True)
    # os.makedirs(new_algorithm_normalized_folder_path, exist_ok=True)
    # os.makedirs(new_algorithm_threshold_folder_path, exist_ok=True)

        # =========================== Process Images ==========================
    results: list[tuple[str, str]] = []
    for image_file in os.listdir(original_image_folder_path):
        image_file_path: str = os.path.join(original_image_folder_path, image_file)
        if is_image_file(image_file_path):
            result: tuple[str, str] = process_image(
                original_image_path=image_file_path,
                # canny_image_folder_path=canny_image_folder_path,
                new_algorithm_folder_path=new_algorithm_folder_path,
                # sigmoid_threshold=sigmoid_threshold,
                # sigmoid_slope_upper_bound=sigmoid_slope_upper_bound,
                # sigmoid_slope_lower_bound=sigmoid_slope_lower_bound,
                # sigmoid_slope_upper_value_x=sigmoid_slope_upper_value_x,
                # sigmoid_slope_lower_value_x=sigmoid_slope_lower_value_x,
                kernel_size=kernel_size,
                image_pixel_output_range=image_pixel_output_range
            )
            results.append(result)


    return results


def main() -> list[tuple[str, str]]:
    """
    主函数，执行图像处理流程。
    该函数设置图像处理的参数，包括图像文件夹路径、Sigmoid阈值、卷积核大小等，然后调用 process_images_in_folder
    函数批量处理图像，并保存卷积结果和Canny边缘检测结果。
    处理完成后，函数将返回一个包含原始图像路径、卷积结果保存路径、阈值处理后的卷积结果保存路径以及Canny边缘检测结果保存路径的元组列表。

    :return: 包含原始图像路径、卷积结果保存路径、阈值处理后的卷积结果保存路径以及Canny边缘检测结果保存路径的元组列表
    """
    # =========================== Parameters ==========================
    image_root_path: str = "./img"
    # sigmoid_threshold: float = 12
    # sigmoid_slope_upper_bound: float = 0.95
    # sigmoid_slope_lower_bound: float = 0.05
    # sigmoid_slope_upper_value_x: float = 20
    # sigmoid_slope_lower_value_x: float = 10
    kernel_size: int = 3   # 单核心，不叠加
    image_pixel_output_range: tuple = (0, 255)

    # =========================== Process Images in Folder ==========================
    results: list[tuple[str, str]] = process_images_in_folder(
        image_root_path=image_root_path,
        # sigmoid_threshold=sigmoid_threshold,
        # sigmoid_slope_upper_bound=sigmoid_slope_upper_bound,
        # sigmoid_slope_lower_bound=sigmoid_slope_lower_bound,
        # sigmoid_slope_upper_value_x=sigmoid_slope_upper_value_x,
        # sigmoid_slope_lower_value_x=sigmoid_slope_lower_value_x,
        kernel_size=kernel_size,
        image_pixel_output_range=image_pixel_output_range
    )

    return results

if __name__ == "__main__":
    main()
