from san_api import SanLandmarkDetector


device = 'cuda'
image_path = './cache_data/cache/test_1.jpg'
model_path = './snapshots/SAN_300W_GTB_itn_cpm_3_50_sigma4_128x128x8/checkpoint_49.pth.tar'
face = (819.27, 432.15, 971.70, 575.87)
det = SanLandmarkDetector(model_path, device)
locs, scores = det.detect(image_path, face)
print(locs,scores)
