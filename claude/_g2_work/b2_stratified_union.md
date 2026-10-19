# B2 人工抽查（按包分层）：hidden_context

合集 _sft0_union：带 hidden_context 词的行 189（按包 {'r1': 68, 'g1': 61, 'g2': 27, 'g3': 33}）；仍以 hidden_context 扣下 1 行。
抽样：仍扣的全放（1），再按包各补 accepted 放行行到 5 条 → 共 20 条（按包 {'r1': 5, 'g1': 5, 'g2': 5, 'g3': 5}）。
判定列由程序填（办法见脚本 docstring：放行词是否真在放行出处的 E 正文 / 观察里、该 E 学生看得到；仍扣词在学生够得着的证据里有没有）。

| # | 包 | run | request | 单元 | 桶 | 词 | 放行出处 | 程序比对 | 判定 | 在 sft0 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | r1 | averimatec_e384_r1_1 | request_0012 | report | held | 1,500 | {} | 1,500: 可见 E 命中 无 / 观察命中 [89] | **对（口径内扣）** | 否 |
| 2 | r1 | averimatec_e414_r1_1 | request_0008 | step | accepted | 1938 | {"1938": "observation@13"} | 1938: observation@13 正文有=True | **对** | 是 |
| 3 | r1 | averimatec_e12_r1_1 | request_0010 | step | accepted | Guatemala | {"Guatemala": "E5"} | Guatemala: E5 正文有=True 学生可见=True | **对** | 是 |
| 4 | r1 | averimatec_e378_r1_1 | request_0012 | report | accepted | October | {"October": "E7"} | October: E7 正文有=True 学生可见=True | **对** | 是 |
| 5 | r1 | averimatec_e402_r1_1 | request_0012 | report | accepted | November, October | {"November": "E8", "October": "E15"} | November: E8 正文有=True 学生可见=True; October: E15 正文有=True 学生可见=True | **对** | 是 |
| 6 | g1 | groundlie360_v46441308_r1g1_1 | request_0008 | step | accepted | Karine, Jean-Pierre | {"Karine": "E15", "Jean-Pierre": "E15"} | Karine: E15 正文有=True 学生可见=True; Jean-Pierre: E15 正文有=True 学生可见=True | **对** | 是 |
| 7 | g1 | fmnv_c_0NqD96FRgOo_r1g1_1 | request_0012 | report | accepted | Party's | {"Party's": "E20"} | Party's: E20 正文有=True 学生可见=True | **对** | 是 |
| 8 | g1 | groundlie360_v38706501_r1g1_1 | request_0012 | report | accepted | Shanghai | {"Shanghai": "E21"} | Shanghai: E21 正文有=True 学生可见=True | **对** | 是 |
| 9 | g1 | fmnv_c_3fafzwHlLB8_r1g1_1 | request_0008 | step | accepted | January, Kenneth | {"January": "E11", "Kenneth": "E11"} | January: E11 正文有=True 学生可见=True; Kenneth: E11 正文有=True 学生可见=True | **对** | 是 |
| 10 | g1 | groundlie360_v39685802_r1g1_1 | request_0009 | step | accepted | Digital, Simulator | {"Digital": "E14", "Simulator": "E14"} | Digital: E14 正文有=True 学生可见=True; Simulator: E14 正文有=True 学生可见=True | **对** | 是 |
| 11 | g2 | fmnv_a_0Uc3vwNlDdY_r1g2_1 | request_0009 | step | accepted | 80.0 | {"80.0": "E18"} | 80.0: E18 正文有=True 学生可见=True | **对** | 是 |
| 12 | g2 | fmnv_a_A5dqLhDAnfQ_r1g2_1 | request_0008 | step | accepted | 14.4 | {"14.4": "observation@14"} | 14.4: observation@14 正文有=True | **对** | 是 |
| 13 | g2 | fmnv_a_fTIOGkfNU6I_r1g2_1 | request_0007 | step | accepted | 50.9 | {"50.9": "observation@14"} | 50.9: observation@14 正文有=True | **对** | 是 |
| 14 | g2 | fmnv_a_B_JVNt3xtGE_r1g2_1 | request_0007 | step | accepted | 36.2 | {"36.2": "observation@22"} | 36.2: observation@22 正文有=True | **对** | 是 |
| 15 | g2 | fmnv_a_0zC0K0YZEzg_r1g2_1 | request_0007 | step | accepted | 27.6 | {"27.6": "observation@22"} | 27.6: observation@22 正文有=True | **对** | 是 |
| 16 | g3 | videosham_4144_v9_r1g3_1 | request_0006 | step | accepted | 4.2 | {"4.2": "observation@11"} | 4.2: observation@11 正文有=True | **对** | 是 |
| 17 | g3 | fmnv_fv_bvim4rsNHkQ_r1g3_1 | request_0009 | step | accepted | 12.6 | {"12.6": "observation@22"} | 12.6: observation@22 正文有=True | **对** | 是 |
| 18 | g3 | averitec_c_1396_r1g3_1 | request_0005 | step | accepted | 1.6 | {"1.6": "observation@13"} | 1.6: observation@13 正文有=True | **对** | 是 |
| 19 | g3 | qvhighlights_611_r1g3_1 | request_0005 | step | accepted | 147.81 | {"147.81": "observation@14"} | 147.81: observation@14 正文有=True | **对** | 否 |
| 20 | g3 | averitec_c_233_r1g3_1 | request_0012 | report | accepted | York | {"York": "E2"} | York: E2 正文有=True 学生可见=True | **对** | 是 |

抽样判定：{'对（口径内扣）': 1, '对': 19}；清单 B2 误扣 ≤ 2。

全部 189 行同法比对（按 包 × 判定）：{'g1/对': 49, 'g1/对（截断依赖，记 B4）': 12, 'g2/对': 20, 'g2/对（截断依赖，记 B4）': 7, 'g3/对': 14, 'g3/对（截断依赖，记 B4）': 19, 'r1/对': 53, 'r1/对（口径内扣）': 1, 'r1/对（截断依赖，记 B4）': 14}

## 抽样里判错的：0 条


## 全部行里判错的（含抽样外）：0 条


## 照口径扣（截断依赖记 B4；或宽口径能找到、export 口径不放：数字没被本步引用 / 只在之后或别的工作区的观察里），列出备查：53 条

- r1 averimatec_e384_r1_1 request_0012 report held/hidden_context:1,500：对（口径内扣）；{"1,500": {"source": null, "export_rule_release": null, "visible_E_hits": [], "observation_hits": [89], "observation_hits_before_same_actor": [], "verdict": "对（口径内扣）"}}；在 sft0=否
- r1 fmnv_c_JltE_BRx7aM_r1_1 request_0005 step held/tool_result_truncated:30.7：对（截断依赖，记 B4）；{"30.7": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- r1 fmnv_c__0oux8o9MJo_r1_1 request_0005 step held/tool_result_truncated:29.6：对（截断依赖，记 B4）；{"29.6": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- r1 fmnv_c_jLK70KY4Zz4_r1_1 request_0005 step held/tool_result_truncated:80.5：对（截断依赖，记 B4）；{"80.5": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- r1 fmnv_c_ue4pMOsllxc_r1_1 request_0005 step held/tool_result_truncated:20.6：对（截断依赖，记 B4）；{"20.6": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- r1 groundlie360_v1929908_r1_1 request_0005 step held/tool_result_truncated:16.2：对（截断依赖，记 B4）；{"16.2": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- r1 groundlie360_v1936701_r1_1 request_0005 step held/tool_result_truncated:109.6：对（截断依赖，记 B4）；{"109.6": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- r1 qvhighlights_603_r1_1 request_0004 step held/tool_result_truncated:111.8：对（截断依赖，记 B4）；{"111.8": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- r1 qvhighlights_8968_r1_1 request_0004 step held/gain0_in_negative_ws：对（截断依赖，记 B4）；{"116.2": {"source": null, "verdict": "对（截断依赖，记 B4）"}, "119": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- r1 qvhighlights_9167_r1_1 request_0004 step held/gain0_in_negative_ws：对（截断依赖，记 B4）；{"82.9": {"source": null, "verdict": "对（截断依赖，记 B4）"}, "112.6": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- r1 videosham_1007_r1_1 request_0003 step held/tool_result_omitted:1.2：对（截断依赖，记 B4）；{"1.2": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- g1 fmnv_a_HhGABZH0LxM_r1g1_1 request_0004 step held/gain0_in_negative_ws：对（截断依赖，记 B4）；{"Andy": {"source": null, "verdict": "对（截断依赖，记 B4）"}, "Cohen": {"source": null, "verdict": "对（截断依赖，记 B4）"}, "Lili": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- g1 fmnv_a_bqPv1PVpT_A_r1g1_1 request_0004 step held/gain0_in_negative_ws：对（截断依赖，记 B4）；{"News": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- g1 fmnv_c_1524680668705402881_r1g1_1 request_0003 step held/gain0_in_negative_ws：对（截断依赖，记 B4）；{"100": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- g1 fmnv_c_3W1DFf3IEFw_r1g1_1 request_0003 step held/gain0_in_negative_ws：对（截断依赖，记 B4）；{"Rachel": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- g1 fmnv_c_3iV1K6tvFIA_r1g1_1 request_0003 step held/tool_result_truncated:Philadelphia：对（截断依赖，记 B4）；{"Philadelphia": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- g1 fmnv_c_jGRROUqrg6U_r1g1_1 request_0003 step held/tool_result_truncated:Instagram：对（截断依赖，记 B4）；{"Instagram": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- g1 videosham_1045_r1g1_1 request_0003 step held/tool_result_omitted:0.3：对（截断依赖，记 B4）；{"0.3": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- g1 videosham_4008_r1g1_1 request_0003 step held/tool_result_omitted:1.5：对（截断依赖，记 B4）；{"1.5": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- g1 videosham_4061_r1g1_1 request_0003 step held/tool_result_truncated:2.2：对（截断依赖，记 B4）；{"2.2": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- g1 videosham_4144_r1g1_1 request_0003 step held/tool_result_omitted:4.2：对（截断依赖，记 B4）；{"4.2": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- g2 fmnv_a_6C0lUMKMj0w_r1g2_1 request_0004 step held/gain0_in_negative_ws：对（截断依赖，记 B4）；{"Olympics": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- g2 fmnv_a_EFMx_tx4r8o_r1g2_1 request_0004 step held/gain0_in_negative_ws：对（截断依赖，记 B4）；{"270": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- g2 fmnv_a__qP0d_xiYTo_r1g2_1 request_0004 step held/gain0_in_negative_ws：对（截断依赖，记 B4）；{"Shively": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- g2 videosham_4037_r1g2_1 request_0003 step held/tool_result_omitted:2.3：对（截断依赖，记 B4）；{"2.3": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- g2 videosham_4066_r1g2_1 request_0004 step held/tool_result_truncated:4.3,0.485：对（截断依赖，记 B4）；{"4.3": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- g2 videosham_4151_r1g2_1 request_0003 step held/tool_result_omitted:2.2：对（截断依赖，记 B4）；{"2.2": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- g3 fmnv_fv_cBWdFgsqyVU_r1g3_1 request_0006 step held/tool_result_truncated:9.7：对（截断依赖，记 B4）；{"9.7": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- g3 qvhighlights_5_r1g3_1 request_0004 step held/tool_result_truncated:128.8：对（截断依赖，记 B4）；{"128.8": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- g3 qvhighlights_604_r1g3_1 request_0004 step held/tool_result_truncated:95.9,110.2：对（截断依赖，记 B4）；{"95.9": {"source": null, "verdict": "对（截断依赖，记 B4）"}, "110.2": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- g3 qvhighlights_605_r1g3_1 request_0004 step held/tool_result_truncated:71.1：对（截断依赖，记 B4）；{"71.1": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- g3 qvhighlights_607_r1g3_1 request_0004 step held/tool_result_truncated:46.4,90.7：对（截断依赖，记 B4）；{"46.4": {"source": null, "verdict": "对（截断依赖，记 B4）"}, "90.7": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- g3 qvhighlights_610_r1g3_1 request_0004 step held/tool_result_truncated:6.9,45.9：对（截断依赖，记 B4）；{"6.9": {"source": null, "verdict": "对（截断依赖，记 B4）"}, "45.9": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- g3 qvhighlights_611_r1g3_1 request_0004 step held/gain0_in_negative_ws：对（截断依赖，记 B4）；{"124.9": {"source": null, "verdict": "对（截断依赖，记 B4）"}, "147.8": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- g3 qvhighlights_612_r1g3_1 request_0004 step held/tool_result_truncated:121.5,145.8：对（截断依赖，记 B4）；{"121.5": {"source": null, "verdict": "对（截断依赖，记 B4）"}, "145.8": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- g3 qvhighlights_6889_r1g3_1 request_0004 step held/gain0_in_negative_ws：对（截断依赖，记 B4）；{"26.4": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- g3 qvhighlights_6890_r1g3_1 request_0004 step held/gain0_in_negative_ws：对（截断依赖，记 B4）；{"109.0": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- g3 qvhighlights_8972_r1g3_1 request_0004 step held/tool_result_truncated:126.2：对（截断依赖，记 B4）；{"126.2": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- g3 videosham_1007_v9_r1g3_1 request_0003 step held/tool_result_omitted:1.2：对（截断依赖，记 B4）；{"1.2": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- g3 videosham_1045_v9_r1g3_1 request_0003 step held/tool_result_omitted:0.3：对（截断依赖，记 B4）；{"0.3": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- g3 videosham_4008_v9_r1g3_1 request_0003 step held/tool_result_omitted:1.5：对（截断依赖，记 B4）；{"1.5": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- g3 videosham_4037_v9_r1g3_1 request_0003 step held/tool_result_omitted:2.3：对（截断依赖，记 B4）；{"2.3": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- g3 videosham_4061_v9_r1g3_1 request_0003 step held/tool_result_omitted:2.2：对（截断依赖，记 B4）；{"2.2": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- g3 videosham_4066_v9_r1g3_1 request_0004 step held/tool_result_truncated:4.3,0.485：对（截断依赖，记 B4）；{"4.3": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- g3 videosham_4144_v9_r1g3_1 request_0003 step held/tool_result_omitted:4.2：对（截断依赖，记 B4）；{"4.2": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- g3 videosham_4151_v9_r1g3_1 request_0003 step held/tool_result_omitted:2.2：对（截断依赖，记 B4）；{"2.2": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- r1 averimatec_l124_r1_1 request_0013 answer held/terminal_major_error:omitted_required_fact：对（截断依赖，记 B4）；{"20962155": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- r1 averimatec_l250_r1_1 request_0013 answer held/tool_result_truncated:9620000083770：对（截断依赖，记 B4）；{"9620000083770": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- r1 averimatec_l716_r1_1 request_0013 answer held/tool_result_truncated:3169956：对（截断依赖，记 B4）；{"3169956": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- r1 fmnv_c_doJ9iSPBUbg_r1_1 request_0014 answer held/tool_result_truncated:Democratic：对（截断依赖，记 B4）；{"Democratic": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- g1 fmnv_a__KneSpxzvFg_r1g1_1 request_0013 answer held/tool_result_truncated:45.7,43.5,4.7：对（截断依赖，记 B4）；{"45.7": {"source": null, "verdict": "对（截断依赖，记 B4）"}, "43.5": {"source": null, "verdict": "对（截断依赖，记 B4）"}, "4.7": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- g1 fmnv_c_1633777204298649602_r1g1_1 request_0013 answer rejected/answer_errors:premature_finish,unsupported_assertion,unsupported_assertion,unsupported_assertion,attribution_drift,unsupported_assertion,unsupported_assertion：对（截断依赖，记 B4）；{"2023": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否
- g2 fmnv_a_d4xTktY_lrw_r1g2_1 request_0011 answer rejected/answer_errors:premature_finish：对（截断依赖，记 B4）；{"39.7": {"source": null, "verdict": "对（截断依赖，记 B4）"}, "57.3": {"source": null, "verdict": "对（截断依赖，记 B4）"}, "0.8": {"source": null, "verdict": "对（截断依赖，记 B4）"}, "14.7": {"source": null, "verdict": "对（截断依赖，记 B4）"}, "141.6": {"source": null, "verdict": "对（截断依赖，记 B4）"}, "142.1": {"source": null, "verdict": "对（截断依赖，记 B4）"}, "22.2": {"source": null, "verdict": "对（截断依赖，记 B4）"}, "44.7": {"source": null, "verdict": "对（截断依赖，记 B4）"}}；在 sft0=否

