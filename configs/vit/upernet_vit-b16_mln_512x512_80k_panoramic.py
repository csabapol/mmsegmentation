_base_ = [
    '../_base_/models/upernet_vit-b16_ln_mln.py',
    '../_base_/datasets/panoramic.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

model = dict(
    pretrained='pretrain/upernet_vit-b16_mln_512x512_80k_ade20k-0403cee1.pth',
    decode_head=dict(num_classes=4),
    auxiliary_head=dict(num_classes=4))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=12)
