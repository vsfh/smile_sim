import numpy as np
from tid_models import YoloModel, KeypointsModel, SegModel, CVModel
from utils import loose_bbox
import asyncio
import cv2


class ProfilePipeline():
    KEYPOINT_NAMES = [
        "G_ts", "N'", "Prn", "Cm", "Sn",
        "A'", "UL'", "Ls", "Stms", "St",
        "Stmi", "Li", "LL'", "Si", "Pog'",
        "Gn'", "Me'", "K", "T", "Or",
        "Bri", "Tr", "G", "Go'", 'Ex', 'neck',
        'middle_1', 'middle_2',
    ]

    LY_KEYPOINT_NAMES = [
        ['G', "N'", 'Or', "Go'", 'T',
         'Prn', 'Cm', 'Sn', "A'", "UL'",
         'Ls', 'Stms', 'Stmi', 'Li', "LL'",
         'Si', "Pog'", "Gn'", "Me'", 'K',
         'Tr', 'Ex', 'neck'],
        ['middle_1', 'middle_2'],
        # ['G_ts'],
    ]

    def __init__(self, backend='native'):

        if backend == 'native':
            self.model_root = 'weights/repository'
        elif backend == 'web':
            self.model_root = "0.0.0.0:8001"
            # self.model_root = "zh-triton.zh-ml-backend.svc:8001"
            

        self.yolo_model = YoloModel(self.model_root, 'face-detector', (512, 512), backend=backend)
        self.kps_model = KeypointsModel(self.model_root, 'face-profile_kps', (384, 640), resize_ratio=0.9,
                                        backend=backend)
        self.show = False

    async def predict_image_async(self, img):
        height, width = img.shape[:2]

        objs = await self.yolo_model.predict_async(img, show=self.show)

        if len(objs[0]) == 0:
            face_bbox = None
        else:
            face_bbox = loose_bbox(objs[0][0], (width, height), 0.15)

        ly_kps = await  self.kps_model.predict_async(img, bbox=face_bbox, show=self.show)
        ly_dict = {k: p for k, p in zip(self.KEYPOINT_NAMES, ly_kps)}

        if self.show:
            self.show_prediction(img, ly_dict)

        kps = {}
        for gid, g in enumerate(self.LY_KEYPOINT_NAMES):
            for pid, n in enumerate(g):
                ly_id = gid * 100 + pid
                kps[f'01{ly_id:04d}'] = ly_dict[n].tolist()

        result = {
            'kps': kps,
            'ly': ly_dict,
        }
        return result

    def show_prediction(self, img, ly_dict):
        height, width = img.shape[:2]
        if self.show:
            vis = img.copy()
            vis_width = 900
            scale = vis_width / width
            vis = cv2.resize(vis, (vis_width, int(height / width * vis_width)))

            for k, (x, y) in ly_dict.items():
                x, y = int(x * scale), int(y * scale)
                cv2.circle(vis, (x, y), 3, (0, 0, 255))
                cv2.putText(vis, k, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0))
            cv2.imshow('img', vis[..., ::-1])
            cv2.waitKey()

class FrontPipeline():
    FACE_KEYPOINT_NAMES = [
        'Ac(L)', 'Ac(R)', 'Ch(L)', 'Ch(R)', 'En(L)', 'En(R)', 'Ex(L)', 'Ex(R)', 'G', "Go'(L)", "Go'(R)",
        'Li', 'Ls', 'Lst', "Me'", 'N', 'O(L)', 'O(R)', 'Pre(L)', 'Pre(R)', 'Prn', 'Pu(L)', 'Pu(R)', 'Si',
        'Sn', 'Sto', 'Tr', 'Ust', 'V1', 'V2', 'Zy(L)', 'Zy(R)', 'buccal_D(L)', 'buccal_D(R)', 'pu1(L)',
        'pu1(R)', 'pu2(L)', 'pu2(R)', 'pu3(L)', 'pu3(R)'
    ]

    LY_KEYPOINT_NAMES = [
        ['Tr', 'G', 'Sn', 'Sto', "Me'",
         'Ch(L)', 'Ch(R)', 'Ac(L)', 'Ac(R)', "Go'(L)", "Go'(R)",
         'Zy(L)', 'Zy(R)', 'Pu(L)', 'Pu(R)', 'Ex(R)', 'En(R)',
         'Ex(L)', 'En(L)', 'Pre(L)', 'Pre(R)', 'Prn', 'Ls', 'Li', 'N',
         'O(L)', 'O(R)'],
        ['V1', 'V2', 'Zero'],
    ]

    YOLO_CLASS_NAMES = ['face', 'tmp', 'mouth', 'nose']

    def __init__(self, backend='native'):
        if backend == 'native':
            self.model_root = 'weights/repository'
        elif backend == 'web':
            self.model_root = '0.0.0.0:8001'
            
            # self.model_root = "zh-triton.zh-ml-backend.svc:8001"

        self.yolo_model = YoloModel(self.model_root, 'face-detector', (512, 512), backend=backend)
        self.face_kps_model = KeypointsModel(self.model_root, 'face-front_kps', (512, 512), resize_ratio=0.9,
                                             backend=backend)
        self.show = False

    def show_prediction(self, img, ly_dict):
        height, width = img.shape[:2]
        vis = img.copy()
        vis_width = 800
        scale = vis_width / width
        vis = cv2.resize(vis, (vis_width, int(height / width * vis_width)))

        a = list(map(int, ly_dict['V1'] * scale))
        b = list(map(int, ly_dict['V2'] * scale))
        cv2.line(vis, a, b, (0, 0, 255), 1)

        for gid, g in enumerate(self.LY_KEYPOINT_NAMES):
            for pid, k in enumerate(g):
                x, y = ly_dict[k]
                x, y = int(x * scale), int(y * scale)
                cv2.circle(vis, (x, y), 3, (0, 0, 255))
                cv2.putText(vis, k, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0))
        cv2.imshow('img', vis[..., ::-1])
        cv2.waitKey()

    def correct_middle_line(self, ly_dict):
        x1, y1 = ly_dict['V1']
        x2, y2 = ly_dict['V2']

        k = (y2 - y1) / (x2 - x1)
        # cx, cy = (ly_dict['Sn'] + ly_dict['Prn']) / 2
        cx, cy = ly_dict['Prn']

        x1 = (y1 - cy) / k + cx
        x2 = (y2 - cy) / k + cx
        ly_dict['V1'] = np.array([x1, y1])
        ly_dict['V2'] = np.array([x2, y2])

    def predict_image(self, img):
        height, width = img.shape[:2]
        objs = self.yolo_model.predict(
            img,
            class_names=self.YOLO_CLASS_NAMES,
            show=self.show,
            class_agnostic=False,
        )

        if 'face' not in objs:
            face_bbox = None
        else:
            face_bbox = loose_bbox(objs['face'][0]['xyxy'], (width, height), 0.15)

        ly_kps = self.face_kps_model.predict(img, bbox=face_bbox, show=self.show)
        ly_dict = {k: p for k, p in zip(self.FACE_KEYPOINT_NAMES, ly_kps)}
        self.correct_middle_line(ly_dict)
        ly_dict['Zero'] = (ly_dict['V2'] + ly_dict['V1']) / 2

        if self.show:
            self.show_prediction(img, ly_dict)

        kps = {}
        for gid, g in enumerate(self.LY_KEYPOINT_NAMES):
            for pid, n in enumerate(g):
                ly_id = gid * 100 + pid
                kps[f'00{ly_id:04d}'] = ly_dict[n].tolist()

        result = {
            'kps': kps,
            'ly': ly_dict
        }
        return result

class SmilePipeLine(FrontPipeline):
    MOUTH_KEYPOINT_NAMES = [
        'Ch(L)', 'Ch(R)', 'buccal_A(L)', 'buccal_A(R)', 'buccal_B(L)', 'buccal_B(R)',
        'm1', 'm2',
        'sl(0)', 'sl(1)', 'sl(2)', 'sl(3)', 'sl(4)', 'sl(5)']


    LY_KEYPOINT_NAMES = [
        ['Tr', 'G', 'Sn', 'Si', "Me'", 'Lst', 'Ust', 'Ch(L)', 'Ch(R)',
         'm1', 'm2',
         'buccal_A(L)', 'buccal_A(R)', 'buccal_B(L)', 'buccal_B(R)',
         'buccal_D(L)', 'buccal_D(R)',
         ],
        ['V1', 'V2', 'Zero'],
        ['sl(0)', 'sl(1)', 'sl(2)', 'sl(3)', 'sl(4)', 'sl(5)'],
        ['13(U)', '13(D)', '13(L)', '13(R)',
         '12(U)', '12(D)', '12(L)', '12(R)',
         '11(U)', '11(D)', '11(L)', '11(R)',
         '21(U)', '21(D)', '21(L)', '21(R)',
         '22(U)', '22(D)', '22(L)', '22(R)',
         '23(U)', '23(D)', '23(L)', '23(R)', ],
        ['Tr', 'G', 'Sn', 'Sto', "Me'",
         'Ch(L)', 'Ch(R)', 'Ac(L)', 'Ac(R)', "Go'(L)", "Go'(R)",
         'Zy(L)', 'Zy(R)', 'Pu(L)', 'Pu(R)', 'Ex(R)', 'En(R)',
         'Ex(L)', 'En(L)', 'Pre(L)', 'Pre(R)', 'Prn', 'Ls', 'Li', 'N',
         'O(L)', 'O(R)',
         'pu1(L)','pu1(R)', 'pu2(L)', 'pu2(R)', 'pu3(L)', 'pu3(R)']
    ]

    def __init__(self, backend):
        super(SmilePipeLine, self).__init__(backend)
        self.mouth_kps_model = KeypointsModel(self.model_root, 'face-mouth_kps', (256, 256), resize_ratio=0.9,
                                              backend=backend)
        self.teeth_model = YoloModel(self.model_root, 'face-teeth_yolo', (256, 256), backend=backend)

    async def predict_teeth_bbox_async(self, img, mouth_rect):
        teeth_objs = await self.teeth_model.predict_async(img, mouth_rect, iou_thr=0.2, class_agnostic=True,
                                                          show=self.show)
        idx = [2, 1, 0, 3, 4, 5]
        names = ['11', '12', '13', '21', '22', '23']

        teeth_kps = {}
        for i in idx:
            objs = teeth_objs[i]
            if len(objs) == 0:
                continue

            x1, y1, x2, y2 = objs[0]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            up = [cx, y1]
            down = [cx, y2]
            left = [x1, cy]
            right = [x2, cy]

            for pt, c in zip([up, down, left, right], ['U', 'D', 'L', 'R']):
                teeth_kps[f'{names[i]}({c})'] = np.array(pt)
        return teeth_kps

    async def predict_image_async(self, img):
        height, width = img.shape[:2]
        objs = await self.yolo_model.predict_async(img, show=self.show)

        if len(objs[0]) == 0 or len(objs[2]) == 0:
            return {'kps': {}}

        face_bbox = loose_bbox(objs[0][0], (width, height), 0.15)
        mouth_bbox = loose_bbox(objs[2][0], (width, height), 0.05)

        x1, y1, x2, y2 = mouth_bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        length = max(w, h)
        x1, y1 = cx - length / 2, cy - length / 2
        x2, y2 = cx + length / 2, cy + length / 2
        mouth_rect = loose_bbox([x1, y1, x2, y2], (width, height), 0.)

        face_kps_coro = self.face_kps_model.predict_async(img, face_bbox)
        mouth_kps_coro = self.mouth_kps_model.predict_async(img, mouth_rect)
        teeth_kps_coro = self.predict_teeth_bbox_async(img, mouth_rect)
        face_kps, mouth_kps, teeth_kps = await asyncio.gather(face_kps_coro, mouth_kps_coro, teeth_kps_coro)

        ly_dict = {k: p for k, p in zip(self.FACE_KEYPOINT_NAMES, face_kps)}
        self.correct_middle_line(ly_dict)
        ly_dict['Zero'] = (ly_dict['V2'] + ly_dict['V1']) / 2

        ly_dict.update(
            {k: p for k, p in zip(self.MOUTH_KEYPOINT_NAMES, mouth_kps)}
        )
        ly_dict.update(teeth_kps)

        if self.show:
            for gid, g in enumerate(self.LY_KEYPOINT_NAMES):
                for pid, n in enumerate(g):
                    ly_id = gid * 100 + pid
                    print(f'02{ly_id:04d}', n, sep='\t')

            self.show_prediction(img, ly_dict)

        kps = {}
        for gid, g in enumerate(self.LY_KEYPOINT_NAMES):
            for pid, n in enumerate(g):
                ly_id = gid * 100 + pid
                if n not in ly_dict:
                    continue

                kps[f'02{ly_id:04d}'] = ly_dict[n].tolist()

        result = {
            'kps': kps
        }
        return result

    def show_prediction(self, img, ly_dict):
        height, width = img.shape[:2]
        vis = img.copy()
        vis_width = 800
        scale = vis_width / width
        vis = cv2.resize(vis, (vis_width, int(height / width * vis_width)))

        a = list(map(int, ly_dict['V1'] * scale))
        b = list(map(int, ly_dict['V2'] * scale))
        cv2.line(vis, a, b, (0, 0, 255), 1)

        a = list(map(int, ly_dict['m1'] * scale))
        b = list(map(int, ly_dict['m2'] * scale))
        cv2.line(vis, a, b, (0, 255, 255), 1)
        for gid, g in enumerate(self.LY_KEYPOINT_NAMES):
            if gid in [2, 3]:
                continue
            for pid, k in enumerate(g):
                if k not in ly_dict:
                    continue

                x, y = ly_dict[k]
                x, y = int(x * scale), int(y * scale)
                cv2.circle(vis, (x, y), 3, (0, 0, 255))
                # if gid not in [2, 3]:
                #     cv2.putText(vis, k, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0))
        cv2.imshow('img', vis[..., ::-1])
        cv2.waitKey()


backend = 'web'
pipeline = FrontPipeline(backend)


        