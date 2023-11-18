from PIL import Image
from torch.utils.data import Dataset
import random
import torch
import numpy as np
from shapely.wkt import loads as shape_loads



CATEGORIES = ["airport", "airport_hangar", "airport_terminal", "amusement_park",
              "aquaculture", "archaeological_site", "barn", "border_checkpoint",
              "burial_site", "car_dealership", "construction_site", "crop_field",
              "dam", "debris_or_rubble", "educational_institution", "electric_substation",
              "factory_or_powerplant", "fire_station", "flooded_road", "fountain",
              "gas_station", "golf_course", "ground_transportation_station", "helipad",
              "hospital", "impoverished_settlement", "interchange", "lake_or_pond",
              "lighthouse", "military_facility", "multi-unit_residential",
              "nuclear_powerplant", "office_building", "oil_or_gas_facility", "park",
              "parking_lot_or_garage", "place_of_worship", "police_station", "port",
              "prison", "race_track", "railway_bridge", "recreational_facility",
              "road_bridge", "runway", "shipyard", "shopping_mall",
              "single-unit_residential", "smokestack", "solar_farm", "space_facility",
              "stadium", "storage_tank", "surface_mine", "swimming_pool", "toll_booth",
              "tower", "tunnel_opening", "waste_disposal", "water_treatment_facility",
              "wind_farm", "zoo"]


def is_invalid_lon_lat(lon, lat):
    return np.isnan(lon) or np.isnan(lat) or \
        (lon in [float('inf'), float('-inf')]) or (lat in [float('inf'), float('-inf')]) or \
        lon < -180 or lon > 180 or lat < -90 or lat > 90


def fmow_temporal_images(example, img_transform, num_frames=3, is_random=False, stack_tensor=True, channel_first=False):
    image_keys = sorted([k for k in example if k.endswith('.npy')])
    metadata_keys = sorted([k for k in example if k.endswith('.json')])
    if len(image_keys) < num_frames:
        while len(image_keys) < num_frames:
            image_keys.append('input-0.npy')
            metadata_keys.append('metadata-0.json')
    else:
        img_md = random.sample(list(zip(image_keys, metadata_keys)), k=num_frames) \
            if is_random else list(zip(image_keys, metadata_keys))[:num_frames]
        image_keys = [img for img, md in img_md]
        metadata_keys = [md for img, md in img_md]

    img = [img_transform(example[k]) for k in image_keys]
    if stack_tensor:
        img = torch.stack(img)  # (T, C, H, W)
        if channel_first:
            img = img.permute(1, 0, 2, 3).contiguous()  # (C, T, H, W)

    return img, metadata_keys


def fmow_numerical_metadata(example, meta_df, target_resolution, num_metadata, rgb_key='input.npy',
                            md_key='metadata.json', base_year=1980, base_lon=180, base_lat=90):
    md = example[md_key]
    h, w, c = example[rgb_key].shape
    assert c == 3, 'Shape error'
    orig_res = min(h, w)

    target_res = target_resolution
    scale = orig_res / target_res
    gsd = md['gsd'] * scale

    cloud_cover = md['cloud_cover'] / 100.

    timestamp = md['timestamp']
    year = int(timestamp[:4]) - base_year
    month = int(timestamp[5:7])
    day = int(timestamp[8:10])
    hour = int(timestamp[11:13])

    # name_items = example['__key__'].split('-')[-1].replace('_rgb', '').split('_')
    name_items = example[md_key]['img_filename'].replace('_rgb', '').replace('.jpg', '').replace('_ms.tif', '').split('_')
    category = CATEGORIES[example['output.cls']] if 'output.cls' in example else example['category.txt']  # eg: recreational_facility
    location_id = int(name_items[-2])  # eg: 890
    image_id = int(name_items[-1])  # eg: 4

    polygon = meta_df[
        (meta_df['category'] == category) &
        (meta_df['location_id'] == location_id) &
        (meta_df['image_id'] == image_id)
    ]['polygon']
    assert len(polygon) == 1, f"{category}, {location_id}, {image_id} is not found in csv"
    poly = shape_loads(polygon.iloc[0])
    lon, lat = poly.centroid.x, poly.centroid.y
    assert not is_invalid_lon_lat(lon, lat)

    return torch.tensor([lon + base_lon, lat + base_lat, gsd, cloud_cover, year, month, day])


def fmow_temporal_preprocess_train(examples, img_transform, num_cond=2, with_target=True, skip_duplicates=False, is_ascending=True, has_dummy_batch=False):
    for example in examples:
        img_temporal, md_keys = fmow_temporal_images(example, img_transform, num_frames=num_cond)

        ascending_idx = np.argsort(md_keys).tolist()
        img_temporal = [img_temporal[i] for i in (ascending_idx if is_ascending else ascending_idx[::-1])]
        md_keys = [md_keys[i] for i in (ascending_idx if is_ascending else ascending_idx[::-1])]

        if has_dummy_batch:
            img_temporal = img_temporal.unsqueeze(0)

        if skip_duplicates and md_keys[0] in md_keys[1:]:  # if target img is in cond, cannot generate
            if with_target:
                yield None, torch.tensor(1)
            else:
                yield None
        else:
            if with_target:
                yield img_temporal, torch.tensor(1)
            else:
                yield img_temporal
