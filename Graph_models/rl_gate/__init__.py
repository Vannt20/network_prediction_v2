"""
RL-Gate: bộ điều phối trọng số các nhánh theo từng bước và từng luồng (spec Mục 6).
- features.py  : dựng trạng thái nhân quả từ cache (chỉ dùng nhãn đến t-1)
- policy.py    : encoder Transformer (lịch sử sai số) + GNN (đồ thị luồng), actor, critic
- env.py       : phần thưởng và mô phỏng phát lại
- sac.py       : Soft Actor-Critic
- supervised.py: gate giám sát (gamma = 0), cùng kiến trúc
"""
