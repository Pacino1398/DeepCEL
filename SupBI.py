'''
Paicno
Sp algorithm after interpolation processing
To support subsequent analysis and evaluation
The code is in the process of being organized...
'''
import argparse
import glob
import numpy as np
import os
import time
import cv2
from numpy.core.records import array
from numpy.distutils.system_info import x11_info
import torch
from data_p.seg_pg import BIREIMG

if int(cv2.__version__[0]) < 3:  
    print('Warning: OpenCV 3 is not installed')


class SuperPointNet(torch.nn.Module):
    def __init__(self):
        super(SuperPointNet, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)  # 编码层
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)  # 解码层
        self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1) 
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        dn = torch.norm(desc, p=2, dim=1) 
        desc = desc.div(torch.unsqueeze(dn, 1)) 
        return semi, desc 


class SuperPointFrontend(object): 
    def __init__(self, weights_path, nms_dist, conf_thresh, nn_thresh,
                 cuda=False):
        self.name = 'SuperPoint'
        self.cuda = cuda
        self.nms_dist = nms_dist
        self.conf_thresh = conf_thresh 
        self.nn_thresh = nn_thresh  
        self.cell = 8  
        self.border_remove = 4 

        self.net = SuperPointNet()
        if cuda:
            self.net.load_state_dict(torch.load(weights_path))
            self.net = self.net.cuda()
        else:
            self.net.load_state_dict(torch.load(weights_path,
                                                map_location=lambda storage, loc: storage))
        self.net.eval()

    def nms_fast(self, in_corners, H, W, dist_thresh):  # 非极大值抑制，dis thresh默认4
        grid = np.zeros((H, W)).astype(int) 
        inds = np.zeros((H, W)).astype(int)  
        inds1 = np.argsort(-in_corners[2, :])
        corners = in_corners[:, inds1]  
        rcorners = corners[:2, :].round().astype(int) 
        if rcorners.shape[1] == 0:
            return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
        if rcorners.shape[1] == 1:
            out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
            return out, np.zeros((1)).astype(int)
        for i, rc in enumerate(rcorners.T):
            grid[rcorners[1, i], rcorners[0, i]] = 1  
            inds[rcorners[1, i], rcorners[0, i]] = i  
        pad = dist_thresh  
        grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant') 
        count = 0
        for i, rc in enumerate(rcorners.T):
            pt = (rc[0] + pad, rc[1] + pad)  
            if grid[pt[1], pt[0]] == 1:  
                grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
                grid[pt[1], pt[0]] = -1 
                count += 1
        keepy, keepx = np.where(grid == -1) 
        keepy, keepx = keepy - pad, keepx - pad  
        inds_keep = inds[keepy, keepx]
        out = corners[:, inds_keep]  
        values = out[-1, :] 
        inds2 = np.argsort(-values)
        out = out[:, inds2]
        out_inds = inds1[inds_keep[inds2]]
        return out, out_inds  

    def run(self, img):
        assert img.ndim == 2, 'Image must be grayscale.'
        assert img.dtype == np.float32, 'Image must be float32.'
        H, W = img.shape[0], img.shape[1]
        inp = img.copy()
        inp = (inp.reshape(1, H, W))
        inp = torch.from_numpy(inp)
        inp = torch.autograd.Variable(inp).view(1, 1, H, W)
        if self.cuda:
            inp = inp.cuda()

        outs = self.net.forward(inp)
        semi, coarse_desc = outs[0], outs[1]
        semi = semi.data.cpu().numpy().squeeze()
        dense = np.exp(semi) 
        dense = dense / (np.sum(dense, axis=0) + .00001) 
        nodust = dense[:-1, :, :]
        Hc = int(H / self.cell)
        Wc = int(W / self.cell)
        nodust = nodust.transpose(1, 2, 0)
        heatmap = np.reshape(nodust, [Hc, Wc, self.cell, self.cell])
        heatmap = np.transpose(heatmap, [0, 2, 1, 3])
        heatmap = np.reshape(heatmap, [Hc * self.cell, Wc * self.cell])
        xs, ys = np.where(heatmap >= self.conf_thresh) 
        if len(xs) == 0:
            return np.zeros((3, 0)), None, None
        pts = np.zeros((3, len(xs)))
        pts[0, :] = ys
        pts[1, :] = xs
        pts[2, :] = heatmap[xs, ys]
        pts, _ = self.nms_fast(pts, H, W, dist_thresh=self.nms_dist) 
        inds = np.argsort(pts[2, :])
        pts = pts[:, inds[::-1]] 
        bord = self.border_remove
        toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
        toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
        toremove = np.logical_or(toremoveW, toremoveH)
        pts = pts[:, ~toremove]
        D = coarse_desc.shape[1]
        if pts.shape[1] == 0:
            desc = np.zeros((D, 0))
        else:
            samp_pts = torch.from_numpy(pts[:2, :].copy())
            samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.)) - 1.
            samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.)) - 1.
            samp_pts = samp_pts.transpose(0, 1).contiguous()
            samp_pts = samp_pts.view(1, 1, -1, 2)
            samp_pts = samp_pts.float()
            if self.cuda:
                samp_pts = samp_pts.cuda()
            desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts)
            desc = desc.data.cpu().numpy().reshape(D, -1)
            desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
        return pts, desc, heatmap


def nn_match_two_way(desc1, desc2, nn_thresh):
    assert desc1.shape[0] == desc2.shape[0]
    if desc1.shape[1] == 0 or desc2.shape[1] == 0:
        return np.zeros((3, 0))
    if nn_thresh < 0.0:
        raise ValueError('\'nn_thresh\' should be non-negative')
    dmat = np.dot(desc1.T, desc2)
    dmat = np.sqrt(2 - 2 * np.clip(dmat, -1, 1))
    idx = np.argmin(dmat, axis=1)
    scores = dmat[np.arange(dmat.shape[0]), idx]
    keep = scores < nn_thresh
    idx2 = np.argmin(dmat, axis=0)
    keep_bi = np.arange(len(idx)) == idx2[idx]
    keep = np.logical_and(keep, keep_bi)
    idx = idx[keep]
    scores = scores[keep]
    m_idx1 = np.arange(desc1.shape[1])[keep]
    m_idx2 = idx
    matches = np.zeros((3, int(keep.sum())))
    matches[0, :] = m_idx1
    matches[1, :] = m_idx2
    matches[2, :] = scores
    return matches


class VideoStreamer(object):
    def __init__(self, basedir, camid, height, width, skip, img_glob):
        self.cap = []  # list
        self.camera = False
        self.video_file = False
        self.listing = []
        self.sizer = [height, width]
        self.i = 0
        self.skip = skip
        self.maxlen = 1000000
        if basedir == "camera/" or basedir == "camera":
            print('==> Processing Webcam Input.')
            self.cap = cv2.VideoCapture(camid)
            self.listing = range(0, self.maxlen)
            self.camera = True
        else:
            self.cap = cv2.VideoCapture(basedir)
            lastbit = basedir[-4:len(basedir)] 
            if (type(self.cap) == list or not self.cap.isOpened()) and (lastbit == '.mp4'):
                raise IOError('Cannot open movie file')
            elif type(self.cap) != list and self.cap.isOpened() and (lastbit != '.txt'):
                print('==> Processing Video Input.')
                num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
                self.listing = range(0, num_frames) 
                self.listing = self.listing[::self.skip] 
                self.camera = True
                self.video_file = True
                self.maxlen = len(self.listing) 
            else:
                print('==> Processing Image Directory Input.')
                search = os.path.join(basedir, img_glob) 
                # type(search)=str
                self.listing = glob.glob(search) 
                self.listing.sort() 
                self.listing = self.listing[::self.skip]
                self.maxlen = len(self.listing)
                if self.maxlen == 0:
                    raise IOError('No images were found (maybe bad \'--img_glob\' parameter?)')

    def read_image(self, impath, img_size): 
        grayim = cv2.imread(impath, 0)
        bgrim = cv2.imread(impath)

        if grayim is None:
            raise Exception('Error reading image %s' % impath)
        interp = cv2.INTER_NEAREST  #   最近邻插值 最优
        grayim = cv2.resize(grayim, (img_size[1], img_size[0]), interpolation=interp)  # 图片缩小到指定HW
        grayim = (grayim.astype('float32') / 255.)

        if bgrim is None:
            raise Exception('Error reading image %s' % impath)
        print(f"BGRImage loaded successfully: {impath}")
        bgrim = cv2.resize(bgrim, (img_size[1], img_size[0]), interpolation=interp)
        return grayim, bgrim

    def next_frame(self):
        """ Return the next frame, and increment internal counter.
        Returns
           image: Next H x W image.
           status: True or False depending whether image was loaded.
        """
        if self.i == self.maxlen: 
            return (None, False, None)
        print('AAAAAAAAAAAAAAAAAAA')
        if self.camera: 
            ret, input_image = self.cap.read()
            raw_img = self.cap.read()
            print('BBBBBBBBBBBBBBBBBBBBB')
            if ret is False:
                print('VideoStreamer: Cannot get image from camera (maybe bad --camid?)')
                return (None, False, None)
            if self.video_file:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.listing[self.i])

            input_image = cv2.resize(input_image, (self.sizer[1], self.sizer[0]),
                                     interpolation=cv2.INTER_AREA)
            raw_img = input_image.copy()
            print('CCCCCCCCCCCCCCCCCCCCCC')
            input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)     #RGB转灰度
            input_image = input_image.astype('float') / 255.0
        else:
            image_file = self.listing[self.i]
            input_image, raw_img  = self.read_image(image_file, self.sizer)
            print('DDDDDDDDDDDDDDDDDDDDD')

        self.i = self.i + 1
        input_image = input_image.astype('float32')
        return (input_image, True, raw_img)


def match_descriptors(kp1, desc1, kp2, desc2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)

    matches_idx = np.array([m.queryIdx for m in matches])
    m_kp1 = [kp1[idx] for idx in matches_idx]
    matches_idx = np.array([m.trainIdx for m in matches])
    m_kp2 = [kp2[idx] for idx in matches_idx]
    return m_kp1, m_kp2, matches


def showpoint(img, ptx):
    if len(img.shape) == 2: 
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i in range(ptx.shape[1]):
        x = int(round(ptx[0, i]))
        y = int(round(ptx[1, i]))

        cv2.circle(img, (x, y), 3, color=(0, 255, 0))
    return img


def drawMatches(img1, kp1, img2, kp2, matches, raw_img1, raw_img2):
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    if len(img1.shape) == 2:
        i1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) == 2:
        i2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    i1 = raw_img1.copy()
    i2 = raw_img2.copy()
    out = np.hstack([i1, i2])
    print("sdsdsd", out.shape)
    for i in range(matches.shape[1]):

        img1_idx = matches[0, i]
        img2_idx = matches[1, i]
        x11 = int(img1_idx)
        y11 = int(img1_idx)
        x22 = int(img2_idx)
        y22 = int(img2_idx)
        x1 = kp1[0, x11]
        y1 = kp1[1, y11]
        x2 = kp2[0, x22]
        y2 = kp2[1, y22]

        a = np.random.randint(0, 256)
        b = np.random.randint(0, 256)
        c = np.random.randint(0, 256)

        cv2.circle(out, (int(np.round(x1)), int(np.round(y1))), 3, (0, 255, 0), 1)  # 圆 半径3
        cv2.circle(out, (int(np.round(x2) + cols1), int(np.round(y2))), 3, (0, 255, 0), 1)

        cv2.line(out, (int(np.round(x1)), int(np.round(y1))), (int(np.round(x2) + cols1), int(np.round(y2))), (a, b, c),
                 1, shift=0)

    return out


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch SuperPoint Demo.')
    parser.add_argument('input', type=str, default='',
                        help='Image directory or movie file or "camera" (for webcam).')
    parser.add_argument('--weights_path', type=str, default='superpoint_v1.pth',
                        help='Path to pretrained weights file (default: superpoint_v1.pth).')
    parser.add_argument('--img_glob', type=str, default='*.png', 
                        help='Glob match if directory of images is specified (default: \'*.png\').')
    parser.add_argument('--skip', type=int, default=1,
                        help='Images to skip if input is movie or directory (default: 1).')
    parser.add_argument('--show_extra', action='store_true',
                        help='Show extra debug outputs (default: False).')
    parser.add_argument('--H', type=int, default=480,
                        help='Input image height (default: 120).')
    parser.add_argument('--W', type=int, default=640,
                        help='Input image width (default:640).')
    parser.add_argument('--display_scale', type=int, default=2,
                        help='Factor to scale output visualization (default: 2).')
    parser.add_argument('--min_length', type=int, default=2,
                        help='Minimum length of point tracks (default: 2).')
    parser.add_argument('--max_length', type=int, default=5,
                        help='Maximum length of point tracks (default: 5).')
    parser.add_argument('--nms_dist', type=int, default=4,
                        help='Non Maximum Suppression (NMS) distance (default: 4).')
    parser.add_argument('--conf_thresh', type=float, default=0.015,
                        help='Detector confidence threshold (default: 0.015).')
    parser.add_argument('--nn_thresh', type=float, default=0.7,
                        help='Descriptor matching threshold (default: 0.7).')
    parser.add_argument('--camid', type=int, default=0,
                        help='OpenCV webcam video capture ID, usually 0 or 1 (default: 0).')
    parser.add_argument('--waitkey', type=int, default=1,
                        help='OpenCV waitkey time in ms (default: 1).')
    parser.add_argument('--cuda', action='store_true',
                        help='Use cuda GPU to speed up network processing speed (default: False)')
    parser.add_argument('--no_display', action='store_true',
                        help='Do not display images to screen. Useful if running remotely (default: False).')
    parser.add_argument('--write', action='store_true',
                        help='Save output frames to a directory (default: False)')
    parser.add_argument('--write_dir', type=str, default='tracker_outputs/',
                        help='Directory where to write output frames (default: tracker_outputs/).')
    opt = parser.parse_args()
    print(opt)

    # 读图
    vs = VideoStreamer(opt.input, opt.camid, opt.H, opt.W, opt.skip, opt.img_glob)

    print('==> Loading pre-trained network.')

    fe = SuperPointFrontend(weights_path=opt.weights_path,
                            nms_dist=opt.nms_dist,  # 非极大值抑制 int距离4
                            conf_thresh=opt.conf_thresh, 
                            nn_thresh=opt.nn_thresh,  # 匹配器阈值0.7
                            cuda=opt.cuda)  
    print('==> Successfully loaded pre-trained network.')

    if not opt.no_display:
        win = 'SuperPoint Tracker'
        cv2.namedWindow(win)
    else:
        print('Skipping visualization, will not show a GUI.')

    font = cv2.FONT_HERSHEY_DUPLEX 
    font_clr = (255, 255, 255)
    font_pt = (4, 12)
    font_sc = 0.4

    if opt.write:
        print('Will write outputs to %s' % opt.write_dir)
        if not os.path.exists(opt.write_dir):
            os.makedirs(opt.write_dir)
    print('Running')

    img1, status, raw_img1 = vs.next_frame()  # 读第一张图
    start1 = time.time()
    pts, desc, heatmap = fe.run(img1)
    end1 = time.time()
    c2 = end1 - start1
    print("第一张图提取用时", c2, "提取特征点数目", pts.shape[1])
    # imgx = img1.copy()
    imgx = raw_img1
    img11 = showpoint(imgx, pts)
    cv2.imshow("imgone", img11)

    img2, status, raw_img2 = vs.next_frame()
    start1 = time.time()
    pts1, desc1, heatmap1 = fe.run(img2)
    end1 = time.time()
    c2 = end1 - start1
    print("第二张图提取用时", c2, "提取特征点数目", pts1.shape[1])
    imgx = raw_img2
    img22 = showpoint(imgx, pts1)
    cv2.imshow("imgtwo", img22)

    match = nn_match_two_way(desc, desc1, 0.7)
    print("图1与图2匹配对数", match.shape[1])

    out = sift_matched_img = drawMatches(img1, pts, img2, pts1, match, raw_img1, raw_img2)
    cv2.namedWindow("matcher", 0)
    cv2.imshow("matcher", out)

    cv2.waitKey(0)
    print('Finshed')

# This file includes code adapted from the SuperPoint project:
# https://github.com/rpautrat/SuperPoint
# Original author: Rémi Pautrat
# License: MIT License
