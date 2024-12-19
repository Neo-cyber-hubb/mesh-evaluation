#create ny neo 2024-9-16 16:39

import numpy as np
from numpy.core.numeric import identity
import open3d as o3d
import os
import argparse
import json
import copy
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap



def read_alignment_transformation(filename):
    with open(filename) as data_file:
        data = json.load(data_file)
    return np.asarray(data["transformation"]).reshape((4, 4))


def write_color_distances(path, pcd, distances, max_distance):
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    # cmap = plt.get_cmap("afmhot")
    # cmap = plt.get_cmap("hot_r")

    # 定义颜色映射中的颜色和位置
    colors_ = ["white", "green", "red"]
    cmap_name = "my_green_white_red"
    n_bins = 100  # 定义颜色映射中的分段数
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors_, N=n_bins)

    distances = np.array(distances)
    colors = cmap(np.minimum(distances, max_distance) / max_distance)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(path, pcd)

def get_f1_score_histo2(threshold,
                        filename_mvs,
                        plot_stretch,
                        distance1,
                        distance2,
                        verbose=True):
    print("[get_f1_score_histo2]")
    dist_threshold = threshold
    if len(distance1) and len(distance2):

        print("len(distance1)")
        print(len(distance1))
        print("len(distance2)")
        print(len(distance2))

        recall = float(sum(d < threshold for d in distance2)) / float(
            len(distance2))
        precision = float(sum(d < threshold for d in distance1)) / float(
            len(distance1))
        fscore = 2 * recall * precision / (recall + precision)
        num = len(distance1)
        bins = np.arange(0, dist_threshold * plot_stretch, dist_threshold / 100)
        hist, edges_source = np.histogram(distance1, bins)
        cum_source = np.cumsum(hist).astype(float) / num

        num = len(distance2)
        bins = np.arange(0, dist_threshold * plot_stretch, dist_threshold / 100)
        hist, edges_target = np.histogram(distance2, bins)
        cum_target = np.cumsum(hist).astype(float) / num

    else:
        precision = 0
        recall = 0
        fscore = 0
        edges_source = np.array([0])
        cum_source = np.array([0])
        edges_target = np.array([0])
        cum_target = np.array([0])

    return [
        precision,
        recall,
        fscore,
        edges_source,
        cum_source,
        edges_target,
        cum_target,
    ]

def EvaluateHisto(
    source,
    target,
    voxel_size,
    threshold,
    filename_mvs,
    plot_stretch,
    scene_name,
    verbose=True,
):
    print("[EvaluateHisto]")
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    s = copy.deepcopy(source)
    s = s.voxel_down_sample(voxel_size) # 为什么要降采样，统一分辨率？
    s.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=100)) # 数量设置与采样率有关

    t = copy.deepcopy(target)
    t = t.voxel_down_sample(voxel_size)
    t.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=100))

    print("[compute_point_cloud_to_point_cloud_distance]")
    # 点云s中每个点到点云t最近点的欧式距离
    distance1 = s.compute_point_cloud_distance(t)
    print("[compute_point_cloud_to_point_cloud_distance]")
    distance2 = t.compute_point_cloud_distance(s)


    source_n_fn = filename_mvs + "/" + scene_name + ".precision.ply"
    target_n_fn = filename_mvs + "/" + scene_name + ".recall.ply"

    print("[ViewDistances] Add color coding to visualize error")
    write_color_distances(source_n_fn, s, distance1, 0.3)     # 为什么*3？

    print("[ViewDistances] Add color coding to visualize error")
    write_color_distances(target_n_fn, t, distance2, 0.3)

    # get histogram and f-score
    [
        precision,
        recall,
        fscore,
        edges_source,
        cum_source,
        edges_target,
        cum_target,
    ] = get_f1_score_histo2(threshold, filename_mvs, plot_stretch, distance1,
                            distance2)
    np.savetxt(filename_mvs + "/" + scene_name + ".recall.txt", cum_target)
    np.savetxt(filename_mvs + "/" + scene_name + ".precision.txt", cum_source)
    np.savetxt(
        filename_mvs + "/" + scene_name + ".prf_tau_plotstr.txt",
        np.array([precision, recall, fscore, threshold, plot_stretch]),
    )

    accuracy = np.sum(distance1) / len(distance1)
    array_dis2 = np.array(distance2)
    filtered_dis2 = array_dis2[array_dis2 < 10 * threshold]
    completeness_filtered = np.sum(filtered_dis2) / len(filtered_dis2)
    completeness = np.sum(distance2) / len(distance2)

    return [
        accuracy,
        completeness,
        completeness_filtered,
        precision,
        recall,
        fscore,
        edges_source,
        cum_source,
        edges_target,
        cum_target,
    ]

def plot_graph(
    scene,
    fscore,
    dist_threshold,
    edges_source,
    cum_source,
    edges_target,
    cum_target,
    plot_stretch,
    mvs_outpath,
    show_figure=False,
):
    f = plt.figure()
    plt_size = [14, 7]
    pfontsize = "medium"

    ax = plt.subplot(111)
    label_str = "precision"
    ax.plot(
        edges_source[1::],
        cum_source * 100,
        c="red",
        label=label_str,
        linewidth=2.0,
    )

    label_str = "recall"
    ax.plot(
        edges_target[1::],
        cum_target * 100,
        c="blue",
        label=label_str,
        linewidth=2.0,
    )

    ax.grid(True)
    plt.rcParams["figure.figsize"] = plt_size
    plt.rc("axes", prop_cycle=cycler("color", ["r", "g", "b", "y"]))
    plt.title("Precision and Recall: " + scene + ", " + "%02.2f f-score" %
              (fscore * 100))
    plt.axvline(x=dist_threshold, c="black", ls="dashed", linewidth=2.0)

    plt.ylabel("# of points (%)", fontsize=15)
    plt.xlabel("Meters", fontsize=15)
    plt.axis([0, dist_threshold * plot_stretch, 0, 100])
    ax.legend(shadow=True, fancybox=True, fontsize=pfontsize)
    # plt.axis([0, dist_threshold*plot_stretch, 0, 100])

    plt.setp(ax.get_legend().get_texts(), fontsize=pfontsize)

    plt.legend(loc=2, borderaxespad=0.0, fontsize=pfontsize)
    plt.legend(loc=4)
    leg = plt.legend(loc="lower right")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.setp(ax.get_legend().get_texts(), fontsize=pfontsize)
    png_name = mvs_outpath + "/PR_{0}_@d_th_0_{1}.png".format(
        scene, "%04d" % (dist_threshold * 10000))
    pdf_name = mvs_outpath + "/PR_{0}_@d_th_0_{1}.pdf".format(
        scene, "%04d" % (dist_threshold * 10000))

    # save figure and display
    f.savefig(png_name, format="png", bbox_inches="tight")
    f.savefig(pdf_name, format="pdf", bbox_inches="tight")
    if show_figure:
        plt.show()

def run_evaluation(gt_dir, re_dir, out_dir, tf_dir, dTau, number_of_points, space_res, plot_stretch, scene):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    ############################  Load GT mesh
    print("Path of GT mesh : %s" % gt_dir)
    # mesh_gt = o3d.io.read_triangle_mesh(source_dir)
    # pcd_gt =  o3d.geometry.TriangleMesh.sample_points_uniformly(mesh_gt, number_of_points)
    # # 可视化采样后的点云
    # o3d.visualization.draw_geometries([pcd_gt])
    # # 保存采样后的点云为 PCD 格式
    # o3d.io.write_point_cloud("GT_pt_sampled.pcd", pcd_gt)

    pcd_gt = o3d.io.read_point_cloud(gt_dir)

    ############################  Load reconstructed mesh
    print("Path of reconstructed mesh : %s" % re_dir)
    mesh_re = o3d.io.read_triangle_mesh(re_dir)
    pcd_re = o3d.geometry.TriangleMesh.sample_points_uniformly(mesh_re, number_of_points)
    # pcd_re = o3d.geometry.TriangleMesh.sample_points_poisson_disk(mesh_re, number_of_points)

    if (tf_dir != ""):
        T = read_alignment_transformation(tf_dir)
        print("transformation matrix: ")
        print(T)
        pcd_re.transform(T)

    # 可视化变换后的点云
    # pcd_gt.paint_uniform_color([1, 0, 0])  # 红色
    # pcd_re.paint_uniform_color([0, 1, 0])  # 绿色
    # o3d.visualization.draw_geometries([pcd_re, pcd_gt])


    ############################  Histogramms and P/R/F1
    [accuracy,
        completeness,
        completeness_filtered,
        precision,
        recall,
        fscore,
        edges_source,
        cum_source,
        edges_target,
        cum_target,] = EvaluateHisto(   pcd_re,
                                        pcd_gt,
                                        space_res,
                                        dTau,
                                        out_dir,
                                        plot_stretch,
                                        scene, )

    eva = [precision, recall, fscore]
    print("==============================")
    print("evaluation result : %s" % scene)
    print("==============================")
    print("distance tau : %.3f" % dTau)
    print("accuracy : %.4f" % accuracy)
    print("completeness : %.4f" % completeness)
    print("completeness_filtered : %.4f" % completeness_filtered)
    print("precision : %.4f" % eva[0])
    print("recall : %.4f" % eva[1])
    print("f-score : %.4f" % eva[2])
    print("==============================")

    # Plotting
    dist_threshold = dTau
    plot_graph(
        scene,
        fscore,
        dist_threshold,
        edges_source,
        cum_source,
        edges_target,
        cum_target,
        plot_stretch,
        out_dir,
    )












if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt_dir",
        type=str,
        required=True,
        help="path to ground truth ply file",
    )
    parser.add_argument(
        "--re_dir",
        type=str,
        required=True,
        help="path to reconstruction ply file",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="output directory, default: an evaluation directory is created in the directory of task_dir",
    )
    parser.add_argument(
        "--transformation_dir",
        type=str,
        default="",
        help= "transformation from reconstructed mesh to GT mesh",
    )
    args = parser.parse_args()

    if args.out_dir.strip() == "":
        args.out_dir = os.path.join(os.path.dirname(args.re_dir), "evaluation")

    dTau = 0.1  # scenes_tau_dict[scene]
    # number_of_points = 6553600  # 均匀采样点云，设置采样点的数量
    # number_of_points = 13254958
    number_of_points = 5580004
    space_res = 0.01
    plot_stretch = 5
    scene = "maicity"

    # 体素降采样的voxel_size越小，结果越好，可能是因为中央区域点密集，准确性好，保留点越多结果越好

    run_evaluation(
        gt_dir=args.gt_dir,
        re_dir=args.re_dir,
        out_dir=args.out_dir,
        tf_dir=args.transformation_dir,
        dTau=dTau,
        number_of_points=number_of_points,
        space_res=space_res,
        plot_stretch=plot_stretch,
        scene=scene
    )
