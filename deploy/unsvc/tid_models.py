import asyncio
import numpy as np
from tid_utils import *
import os
from typing import Optional, Dict, List, Tuple
import cv2


class BaseModel():
    def __init__(self, model_root, model_name, version=1, backend='native'):
        assert backend in ['native', 'web']
        self.model_root = model_root
        self.model_name = model_name
        self.backend = backend

        if self.backend == 'native':
            import onnxruntime
            sess_option = onnxruntime.SessionOptions()
            # sess_option.log_severity_level = 4

            path = os.path.join(self.model_root, self.model_name, str(version), 'model.onnx')
            run_options = onnxruntime.RunOptions()
            # run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:0;cpu:0")
            self.sess = onnxruntime.InferenceSession(path, sess_options=sess_option,
                                                     run_options=run_options,
                                                     providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        else:
            # import tritoninferencer as tinf
            from tritoninferencer import TritonInferencer

            # self.client = TritonInferencer("zh-triton.zh-ml-backend.svc:8001")
            self.client = TritonInferencer("127.0.0.1:8001")
            
            


    def inference(self, input_feed):
        if not isinstance(input_feed, dict):
            raise TypeError('input feed must be a dictionary')

        if self.backend == 'native':
            result = self.sess.run([], input_feed)
            outputs = {}
            for i, node in enumerate(self.sess.get_outputs()):
                outputs[node.name] = result[i]

        elif self.backend == 'web':
            import tritoninferencer as tinf
            # outputs = tinf.infer(self.model_name, input_feed, self.client)
            outputs = self.client.infer_sync(self.model_name, input_feed, ['bbox','seg','bin1','bin2','proto'])

        return outputs

    async def inference_async(self, input_feed, delay_time=0.1, max_iters=1000):
        if self.backend == 'native':
            # fake I/O simulation
            await asyncio.sleep(0.05)
            outputs = self.inference(input_feed)

        else:
            import tritoninferencer as tinf
            outputs = {}
            tinf.infer_async(0, self.model_name, input_feed, outputs, self.client)

            for i in range(max_iters):
                await asyncio.sleep(delay_time)
                if len(outputs) > 0:
                    outputs = outputs[0]
                    if isinstance(outputs, str):
                        print(outputs)
                    break
            else:
                from zhmle.workflow.common import TaskException
                raise TaskException(
                    reason="triton model {} did not return in {} seconds".format(
                        self.model_name,
                        max_iters * delay_time),
                    reason_public="algorithm execution timeout")

            if isinstance(outputs, str):
                from zhmle.workflow.common import TaskException
                raise TaskException(
                    reason="triton model {} failed: {}".format(
                        self.model_name,
                        outputs),
                    reason_public="algorithm execution failed")

        return outputs


class CVModel(BaseModel):
    def __init__(self, model_root, model_name, image_size,
                 resize_ratio=1.,
                 mean=0,
                 std=255,
                 align_mode='center',
                 input_name='images',
                 output_map=None,
                 post_params: Optional[Dict[str, List[str]]] = None,
                 pad_val=0,
                 version=1,
                 backend='native'):
        super(CVModel, self).__init__(model_root, model_name, version, backend)
        self.image_size = image_size
        self.resize_ratio = resize_ratio
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.align_mode = align_mode
        self.pad_val = pad_val

        self.input_name = input_name
        self.output_map = output_map
        if post_params is None:
            self.post_params = {'default': {}}
        self.init_post_params()

    def init_post_params(self):
        pass

    def predict(self, img, bbox=None, show=False, **kwargs):
        input_feed, meta = self.preprocess(img, bbox, **kwargs)
        outputs = self.inference(input_feed)

        if self.output_map is not None:
            mapped_output = {}
            for k, v in outputs.items():
                if k in self.output_map:
                    k = self.output_map[k]
                mapped_output[k] = v
                outputs = mapped_output

        results = []
        for post_type, params in self.post_params.items():
            post_func = getattr(self, f'{post_type}_postprocess', None)
            if post_func is None:
                raise NotImplementedError(f'postprocess {post_type} not implemented')
            show_func = getattr(self, f'{post_type}_vis', None)
            if show_func is None:
                show_func = self.default_vis

            if params is None or len(params) == 0:
                post_inputs = outputs
            else:
                post_inputs = {}
                for k, v in params.items():
                    post_inputs = {k: outputs[v] for k in params}

            post_outputs = post_func(post_inputs, meta, **kwargs)
            results.append(post_outputs)

            if show:
                show_func(img, post_outputs)

        if len(results) == 1:
            return results[0]
        else:
            return {k: v for k, v in zip(self.post_params, results)}

    async def predict_async(self, img, bbox=None, show=False, **kwargs):
        input_feed, meta = self.preprocess(img, bbox, **kwargs)
        outputs = await self.inference_async(input_feed)

        if self.output_map is not None:
            mapped_output = {}
            for k, v in outputs.items():
                if k in self.output_map:
                    k = self.output_map[k]
                mapped_output[k] = v
                outputs = mapped_output

        results = []
        for post_type, params in self.post_params.items():
            post_func = getattr(self, f'{post_type}_postprocess', None)
            if post_func is None:
                raise NotImplementedError(f'postprocess {post_type} not implemented')
            show_func = getattr(self, f'{post_type}_vis', None)
            if show_func is None:
                show_func = self.default_vis

            if params is None or len(params) == 0:
                post_inputs = outputs
            else:
                post_inputs = {}
                for k, v in params.items():
                    post_inputs = {k: outputs[v] for k in params}

            post_outputs = post_func(post_inputs, meta, **kwargs)
            results.append(post_outputs)

            if show:
                show_func(img, post_outputs)

        if len(results) == 1:
            return results[0]
        else:
            return {k: v for k, v in zip(self.post_params, results)}

    def preprocess(self, img, bbox, **kwargs):
        input_img, meta = preprocess_image(
            img,
            self.image_size,
            self.resize_ratio, bbox,
            align_mode=self.align_mode,
            pad_val=self.pad_val,
        )

        input_tensor = blob_image(input_img, self.mean, self.std)
        input_feed = {self.input_name: input_tensor}
        return input_feed, meta

    # default postprocess
    def default_postprocess(self, outputs, meta, **kwargs):
        return outputs, meta

    def default_vis(self, img, result):
        return

    @staticmethod
    def yolo_postprocess(outputs, meta,
                         score_thr=0.2,
                         iou_thr=0.4,
                         class_agnostic=True,
                         class_names=None,
                         loose_factor=0.,
                         magic_str='',  # only used for temporary compatibility, will be removed later
                         smooth=False,
                         return_mask=False,
                         ):
        # for compatibility, will be removed later
        if magic_str == 'mooeli':
            out = outputs['output']
            grid = np.concatenate([out[..., :4], out[..., 6:6 + 38]], axis=-1)
            outputs = {
                'output': grid,
                'angle': out[:, :, 4:6],
                'reg': out[:, :, -6:]
            }

        grid = outputs.pop('output')[0]
        if 'proto' in outputs:
            proto = outputs.pop('proto')[0]
        else:
            proto = None

        yolo_cxywh = grid[:, :4]
        conf = grid[:, 4:5]
        pcls = grid[:, 5:]
        probs = conf * pcls

        labels = np.argmax(probs, axis=1)
        scores = np.max(probs, axis=1)

        num_objs, num_classes = probs.shape[:2]
        offsets = np.array(meta['offsets'])
        scale = meta['scale']
        rect = np.array(meta['rect'])

        yolo_xyxy = np.zeros_like(yolo_cxywh)
        yolo_xyxy[:, 0] = yolo_cxywh[:, 0] - yolo_cxywh[:, 2] / 2
        yolo_xyxy[:, 1] = yolo_cxywh[:, 1] - yolo_cxywh[:, 3] / 2
        yolo_xyxy[:, 2] = yolo_cxywh[:, 0] + yolo_cxywh[:, 2] / 2
        yolo_xyxy[:, 3] = yolo_cxywh[:, 1] + yolo_cxywh[:, 3] / 2

        # map back
        image_xyxy = yolo_xyxy.copy()
        image_xyxy[:, [0, 2]] -= offsets[0]
        image_xyxy[:, [1, 3]] -= offsets[1]
        image_xyxy /= scale
        image_xyxy[:, [0, 2]] += meta['rect'][0]
        image_xyxy[:, [1, 3]] += meta['rect'][1]

        # remove invalid bbox first
        width, height = meta['image_size']
        image_xyxy[:, [0, 2]] = np.clip(image_xyxy[:, [0, 2]], 0, width)
        image_xyxy[:, [1, 3]] = np.clip(image_xyxy[:, [1, 3]], 0, height)

        min_length = 4
        # mx, my = width / 10, height / 10
        x1, y1, x2, y2 = image_xyxy[:, 0], image_xyxy[:, 1], image_xyxy[:, 2], image_xyxy[:, 3]
        invalid_index = (x1 >= (width - 1 - min_length)) | (y1 >= (height - 1 - min_length)) \
                        | (x2 <= min_length) | (y2 <= min_length) | \
                        ((x1 + min_length) >= x2) | ((y1 + min_length) >= y2)

        invalid_index = ((x1 + min_length) >= x2) | ((y1 + min_length) >= y2)
        scores[invalid_index] = 0.

        image_xywh = image_xyxy.copy()
        image_xywh[:, 2] = image_xyxy[:, 2] - image_xyxy[:, 0]
        image_xywh[:, 3] = image_xyxy[:, 3] - image_xyxy[:, 1]

        if class_agnostic:
            keep = cv2.dnn.NMSBoxes(image_xywh, scores, score_thr, iou_thr)
        else:
            keep = []
            number = np.arange(num_objs)
            for i in range(num_classes):
                indices = labels == i
                bboxes = image_xywh[indices]
                cur_number = number[indices]
                cur_scores = scores[indices]
                cur_keep = cv2.dnn.NMSBoxes(bboxes, cur_scores, score_thr, iou_thr)
                if len(cur_keep) != 0:
                    keep.extend(cur_number[cur_keep])

        objs = [[] for _ in range(num_classes)]

        objs = {}
        for k in keep:
            l = labels[k]
            s = scores[k]

            if class_names is None:
                n = l
            else:
                n = class_names[l]

            cxywh = xyxy2cxywh(image_xyxy[k])
            bbox = {
                'cxywh': cxywh,
                'xyxy': verify_xyxy(image_xyxy[k], meta['image_size']).astype(int),  # for convenience
                'prob': float(s),
            }

            for head, v in outputs.items():
                v = v[0]
                if head == 'angle':
                    bbox['angle'] = v[k]
                elif head == 'coefficient':
                    assert proto is not None

                    dst_shape = np.array(meta['dst_shape'])
                    pheight, pwidth = proto.shape[1:]
                    proto_shape = np.array((pwidth, pheight))

                    xyxy = cxywh2xyxy(yolo_cxywh[k])
                    stride = dst_shape / proto_shape
                    xyxy[[0, 2]] = xyxy[[0, 2]] / stride[0]
                    xyxy[[1, 3]] = xyxy[[1, 3]] / stride[1]

                    xyxy[[0, 1]] = np.ceil(xyxy[[0, 1]])
                    xyxy[[2, 3]] = np.floor(xyxy[[2, 3]]) + 1

                    xyxy = verify_xyxy(xyxy, proto_shape)
                    x1, y1, x2, y2 = loose_bbox(xyxy, (pwidth, pheight), loose_factor)

                    coef = v[k]
                    mask = np.sum(coef[:, None, None] * proto[:, y1:y2, x1:x2], axis=0)

                    def sigmoid(z):
                        return 1 / (1 + np.exp(-z))

                    mask = sigmoid(mask)
                    h, w = mask.shape[:2]
                    mask = cv2.resize(mask, (int(w * stride[0]), int(h * stride[1])), cv2.INTER_LINEAR)
                    mask = (mask > 0.5).astype(np.uint8)

                    contours = extract_shapes_from_mask(mask, kernel_size=3, smooth=smooth)
                    points = []
                    for i in range(len(contours)):
                        p = contours[i]
                        if p is None:
                            continue
                        p = (p + np.array((x1, y1)) * stride)
                        p = (p - offsets[:2]) / scale + rect[:2]
                        points.append(p)

                    bbox['points'] = points

                elif head.startswith('reg'):
                    bbox['reg'] = v[k]

                else:
                    bbox[head] = v[k]

            if n not in objs:
                objs[n] = []
            objs[n].append(bbox)
        return objs

    def yolo_vis(self, img, result):
        objs = result
        img = img.copy()

        height, width = img.shape[:2]
        show_length = 1200
        scale = show_length / width
        img = cv2.resize(img, (show_length, int(scale * height)))
        for k, bboxes in objs.items():
            for bbox in bboxes:
                cx, cy, w, h = bbox['cxywh']
                x1, y1 = cx - w / 2, cy - h / 2
                x2, y2 = cx + w / 2, cy + h / 2

                xyxy = np.array([x1, y1, x2, y2])
                x1, y1 = np.ceil(xyxy[[0, 1]] * scale).astype(int)
                x2, y2 = np.floor(xyxy[[2, 3]] * scale).astype(int) + 1

                if 'points' in bbox:
                    pts = bbox['points']
                    for p in pts:
                        p = np.round(p * scale)
                        cv2.polylines(img, [p[:, None].astype(int)], True, (255, 255, 0))

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(img, str(k), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 0, 255))


    # keypoints
    @staticmethod
    def kps_postprocess(outputs, meta, mode='single', conf=0.6, dist=0.04):
        assert mode in ['single', 'nms']

        image_size = np.array(meta['dst_shape'])
        if mode == 'single':
            kps = outputs['kps'][0].copy()

            kps *= image_size

            offset = meta['offsets']
            scale = meta['scale']

            kps[:, 0] -= offset[0]
            kps[:, 1] -= offset[1]
            kps /= scale
            kps += np.array(meta['rect'][:2])
        else:
            heatmaps = outputs['heatmaps'][0]

            c, h, w = heatmaps.shape
            diag_len = np.sqrt(h * h + w * w)
            kps = []
            for i in range(c):
                flat = heatmaps[i].flatten()
                grid = np.arange(h * w)
                probs = flat[flat > conf]
                cands = grid[flat > conf]

                sorted_idx = np.argsort(probs)[::-1]
                pts = []
                for idx in sorted_idx:
                    y, x = cands[idx] / w, cands[idx] % w
                    p1 = np.array([x, y])
                    for j, pt in enumerate(pts):
                        p = pt[0][0]
                        d = np.linalg.norm(p1 - p) / diag_len
                        if d < dist:
                            pts[j].append((p1, probs[idx]))
                            break
                    else:
                        pts.append([(p1, probs[idx])])
                        # pts.append(p1)

                for i, pt in enumerate(pts):
                    avg_p = 0
                    total_prob = 0
                    for p, prob in pt:
                        avg_p += p * prob
                        total_prob += prob
                    pts[i] = avg_p / total_prob

                if len(pts) == 0:
                    kps.append([])
                else:
                    pts = np.array(pts)
                    offset = meta['offsets']
                    scale = meta['scale']

                    pts *= 2
                    pts[:, 0] -= offset[0]
                    pts[:, 1] -= offset[1]
                    pts /= scale

                    pts += np.array(meta['rect'][:2])

                    kps.append(pts)

        return kps

    def kps_vis(self, img, results):
        if not isinstance(results, list):
            kps = [results]
        else:
            kps = results

        img = img.copy()
        height, width = img.shape[:2]
        scale = 800 / width
        img = cv2.resize(img, (800, int(scale * height)))

        count = 0
        for pts in kps:
            for (x, y) in pts:
                x = int(x * scale)
                y = int(y * scale)
                cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), 1)
                cv2.putText(img, str(count), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
                count += 1

        return

    # seg
    def seg_postprocess(self, outputs, meta, keep=False):
        width, height = meta['image_size']
        mask = np.zeros((height, width), dtype=np.uint8)

        segmap = outputs['outputs'][0]
        segmap = segmap.astype(np.uint8)
        x1, y1, x2, y2 = meta['offsets']

        segmap = segmap[y1:y2, x1:x2]
        x1, y1, x2, y2 = meta['rect']
        segmap = cv2.resize(segmap, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
        mask[y1:y2, x1:x2] = segmap
        return mask

    def seg_vis(self, img, result):
        mask = result


    # instance
    @staticmethod
    def instance_postprocess(outputs, meta,
                             score_threshold=0.2,
                             nms_threshold=0.7,
                             smooth=True,
                             grid_sample=False,
                             class_agnostic=False,
                             class_names=None,
                             ):

        dets = outputs['dets'][0]
        labels = outputs['labels'][0]
        masks = outputs['masks'][0].copy()

        probs = dets[:, 4]
        xyxy = np.array(dets[:, :4])

        width, height = meta['image_size']
        xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, width)
        xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, height)

        labels_set = set(labels.tolist())
        num = len(dets)

        new_xywh = np.zeros_like(xyxy)
        points_list = []
        offsets = meta['offsets']
        scale = meta['scale']

        mh, mw = masks.shape[1:]
        for i in range(num):
            if probs[i] < score_threshold:
                points_list.append(None)
            else:
                points = extract_shapes_from_mask(masks[i], num=1, smooth=smooth)[0]
                if points is None:
                    points_list.append(None)
                    continue

                pts = points.astype(np.float32)
                if not grid_sample:
                    x1, y1, x2, y2 = xyxy[i]
                    w, h = x2 - x1, y2 - y1
                    if w <= 0 or h <= 0:
                        continue

                    pts[:, 0] = pts[:, 0] / mw * w
                    pts[:, 1] = pts[:, 1] / mh * h
                    pts += np.array([x1, y1])

                pts -= np.array(offsets[:2])
                pts /= scale
                pts += np.array(meta['rect'][:2])

                x1, y1, x2, y2 = pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()
                new_xywh[i] = (x1, y1, x2 - x1, y2 - y1)
                points_list.append(pts)

        # result = matrix_nms(masks, labels, probs)
        # keep = [i for i,r in enumerate(result) if r > 0.4]

        if class_agnostic:
            keep = cv2.dnn.NMSBoxes(new_xywh, probs, score_threshold, nms_threshold)
        else:
            keep = []
            for l in labels_set:
                indices = labels == l
                bboxes = new_xywh[indices]
                number = np.arange(num)[indices]
                scores = probs[indices]
                cur_keep = cv2.dnn.NMSBoxes(bboxes, scores, score_threshold, nms_threshold)
                if len(cur_keep) != 0:
                    keep.extend(number[cur_keep])

        post_result = {}
        for k in keep:
            l = labels[k]
            if class_names is None:
                n = l
            else:
                n = class_names[l]

            p = probs[k]
            pts = points_list[k]
            if pts is None:
                continue

            x1, y1 = np.min(pts, axis=0)
            x2, y2 = np.max(pts, axis=0)

            if n not in post_result:
                post_result[n] = []
            post_result[n].append({
                'points': pts,
                'prob': float(p),
                'xyxy': [x1, y1, x2, y2],
            })
        return post_result

    def instance_vis(self, img, result):
        post_output = result
        vis = np.ascontiguousarray(img.copy())
        for k, v in post_output.items():
            for info in v:
                pts = info['points']
                x1, y1 = pts.mean(axis=0)

                pts = np.array(pts, dtype=int)[:, None, :]
                cv2.drawContours(vis, [pts], -1, (0, 0, 255), 1)

                cv2.putText(vis, str(k), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 0, 255))


class YoloModel(CVModel):
    def init_post_params(self):
        self.post_params = {'yolo': {}}


class KeypointsModel(CVModel):
    def init_post_params(self):
        self.post_params = {'kps': {}}


class SegModel(CVModel):
    def init_post_params(self):
        self.post_params = {'seg': {}}


class ClsModel(BaseModel):
    def __init__(self, model_root, model_name, image_size, resize_ratio=1., type='native'):
        super(ClsModel, self).__init__(model_root, model_name, type)
        self.image_size = image_size
        self.resize_ratio = resize_ratio

    def predict(self, img, bbox=None, show=False):
        pass


class InstanceModel(CVModel):
    def init_post_params(self):
        self.post_params = {'instance': {}}
        
def get_yolo_tid(model_root,backend = 'web'):
    teeth_model = YoloModel(model_root, 'new_smile_tid', (256, 256),
                            backend=backend, pad_val=114,
                            output_map={
                                'bbox': 'output',
                                'seg': 'coefficient',
                            },
                            align_mode='center')
    return teeth_model

def get_tid(teeth_model, img):
    result = teeth_model.predict(img, show=False)

    img_show = np.zeros((256,256))
    upper = 11
    lower = 33
    points_upper = -np.inf
    points_lower = -np.inf
    area_dict = {}
    for res in result[0]:
        if len(res['points'])==0:
            continue
        points = res['points'][0]
        bin1 = res['bin1']
        bin2 = res['bin2']

        a = np.argmax(bin1) + 1
        b = np.argmax(bin2) + 1
        fdi = int(a*10+b)
        contour = points.reshape((-1, 1, 2)).astype(np.int32)
        area = cv2.contourArea(contour)
        area_dict[fdi] = area
        if area> points_upper and a in [1]:
            upper = fdi
            points_upper = area
        if area> points_lower and a in [3]:
            lower = fdi
            points_lower = area
    if upper==11 and lower==31:
        lower = 33
        
    for res in result[0]:
        if len(res['points'])==0:
            continue
        points = res['points'][0]
        bin1 = res['bin1']
        bin2 = res['bin2']

        a = np.argmax(bin1) + 1
        b = np.argmax(bin2) + 1
        fdi = int(a*10+b)
        if fdi==upper or fdi==lower:
            cv2.fillPoly(img_show, pts=[points.astype(int)[:, None]], color=(1))
    return img_show, upper, lower
 
camera_dict= {
    'z_init':400,
    'z_init_width':33,
    'z_change':7/90,
    'y_init_11_low':144,
    'y_change':4,
    'y_init_31_up':137
}

def mask_width(mask):
    bg = 1e6
    lin = np.concatenate([x * np.ones((256,1)) for x in np.linspace(0, 255, 256)], 1)
    lin_mask = lin * mask
    lower_mask = np.max(lin_mask, 1)

    lin_mask[lin_mask==0]=bg
    upper_mask = np.min(lin_mask, 1)
    upper_mask[upper_mask==bg] = 0
    return upper_mask, lower_mask

def mask_length(mask):
    bg = 1e6
    lin = np.concatenate([x * np.ones((1,256)) for x in np.linspace(0, 255, 256)], 0)
    lin_mask = lin * mask
    lower_mask = np.max(lin_mask, 0)

    lin_mask[lin_mask==0]=bg
    upper_mask = np.min(lin_mask, 0)
    upper_mask[upper_mask==bg] = 0
    return upper_mask, lower_mask
               
def parameter_pred(mouth, edgeu, edged, model):

    threshold = [3, None, None]

    tid = get_tid(model, mouth)

    if np.sum(edgeu)<threshold[0]:
        pass
    if np.sum(edged)<threshold[0]:
        pass
    # if np.sum(mask)<threshold[0]:
    #     pass
    # mask_up, mask_down = mask_length(mask)
    # if np.sum(mask_down-mask_up)<threshold[0]:
    #     pass
    edgeu_up, edgeu_down = mask_length(edgeu)
    edged_up, edged_down = mask_length(edged)
    mask_11 = np.zeros_like(tid)
    mask_11[tid==11]=1

    if np.sum(mask_11)<10:
        mask_21 = np.zeros_like(tid)
        mask_21[tid==21]=1
        if np.sum(mask_21) < 10:
            return 0,0,470,0
        else:
            teeth_mask = mask_21
    else:
        teeth_mask = mask_11


    left_11, right_11 = mask_width(teeth_mask)
    up_11, down_11 = mask_length(teeth_mask)

    pos_left_11 = left_11[left_11>0]
    pos_right_11 = right_11[right_11>0]
    width_11 = np.mean(pos_right_11[int(len(pos_right_11)/3):int(len(pos_right_11)/3*2)])\
               -np.mean(pos_left_11[int(len(pos_left_11)/3):int(len(pos_left_11)/3*2)])
    # width_11 = np.max(right_11-left_11)
    width_11 = width_11.clip(0,35)

    camera_z = camera_dict['z_init']-(width_11-camera_dict['z_init_width']-3)/camera_dict['z_change']
    camera_y = (np.mean(down_11[down_11>0])-camera_dict['y_init_11_low']-(width_11-camera_dict['z_init_width'])/2)/camera_dict['y_change']
    if np.sum(mask_11)<10:
        camera_x = ((np.max(left_11))-127)/camera_dict['y_change']
    else:
        camera_x = ((np.max(right_11))-127)/camera_dict['y_change']

    camera_x = camera_x.clip(-2.5,2.5)
    pos_edgeu_down = edgeu_down[edgeu_down>0]
    pos_edged_up = edged_up[edged_up>0]
    dist = np.mean(pos_edged_up[int(len(pos_edged_up)/3):int(len(pos_edged_up)/3*2)])\
               -np.mean(pos_edgeu_down[int(len(pos_edgeu_down)/3):int(len(pos_edgeu_down)/3*2)])

    if dist>threshold[0]:
        dist = dist+camera_dict['y_init_11_low']-camera_dict['y_init_31_up']
        dist_lower = dist/camera_dict['y_change']
    else:
        dist_lower = 0
    return camera_x, camera_y, camera_z, dist_lower
