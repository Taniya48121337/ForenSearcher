# S0–S3 教师并排（只报数）

## query 组（每边 20 行；层 → 行数 {"C|S_text_text": 20}）

| 指标 | codex | qwen |
|---|---|---|
| rows | 20 | 20 |
| query_rows | 20 | 20 |
| query_words_mean | 10.85 | 8.45 |
| query_claim_share_mean | 0.234 | 0.394 |
| copies_claim | 0/20 | 3/20 |
| page_rationale_rows | 0 | 0 |
| page_rationale_words_mean | None | None |
| rationale_words_mean | 25.3 | 22.9 |
| rationale_arg_overlap_mean | 0.284 | 0.535 |

## page 组（每边 20 行；层 → 行数 {"C|R_page": 8, "C|R_page_more": 3, "C|S_dates": 8, "C|S_open_video": 1}）

| 指标 | codex | qwen |
|---|---|---|
| rows | 20 | 20 |
| query_rows | 0 | 0 |
| query_words_mean | None | None |
| query_claim_share_mean | None | None |
| copies_claim | None | None |
| page_rationale_rows | 20 | 20 |
| page_rationale_words_mean | 25.6 | 21.2 |
| rationale_words_mean | 25.6 | 21.2 |
| rationale_arg_overlap_mean | 0.95 | 0.95 |

## action 组（每边 20 行；层 → 行数 {"B|I_asr": 2, "B|I_compare_video": 2, "B|I_frames_box": 1, "B|I_frames_times": 1, "B|I_locate": 1, "B|I_ocr_image": 1, "C|I_asr": 1, "C|I_compare_video": 1, "C|I_frames_times": 1, "C|I_ocr_video": 1, "C|I_video_structure": 1, "C|R_frames_review": 1, "C|R_page": 1, "C|R_page_more": 1, "C|R_record": 1, "C|S_dates": 1, "C|S_open_video": 1, "C|S_text_text": 1}）

| 指标 | codex | qwen |
|---|---|---|
| rows | 20 | 20 |
| query_rows | 1 | 1 |
| query_words_mean | 15.0 | 11.0 |
| query_claim_share_mean | 0.0 | 0.545 |
| copies_claim | 0/1 | 0/1 |
| page_rationale_rows | 6 | 6 |
| page_rationale_words_mean | 32.833 | 30.333 |
| rationale_words_mean | 40.35 | 40.6 |
| rationale_arg_overlap_mean | 0.753 | 0.823 |

## 全体（不抽样）

| 指标 | codex | qwen |
|---|---|---|
| rows | 1224 | 1221 |
| query_rows | 216 | 96 |
| query_words_mean | 10.722 | 7.771 |
| query_claim_share_mean | 0.214 | 0.372 |
| copies_claim | 6/216 | 15/96 |
| page_rationale_rows | 465 | 621 |
| page_rationale_words_mean | 30.763 | 35.638 |
| rationale_words_mean | 35.693 | 40.043 |
| rationale_arg_overlap_mean | 0.83 | 0.911 |

## 并排（同组同层同序）

### query
- **codex** `C|S_text_text` groundlie360_v46443403_r1g1_1 request_0010：`{"input_type": "text", "query": "site.president.gov.ua 24 February 2023 press conference America NATO Baltic states sons daughters", "output_type": "text"}` — R4–R8 could not be opened, so a targeted search for the presidential account may establish the original context of the disputed statement.
- **qwen** `C|S_text_text` fmnv_c_TIu19O02X2c_r1_1 request_0006：`{"input_type": "text", "query": "SpaceX Starship flight test January 16 2025 explosion airlines resume operations", "output_type": "text"}` — The video shows Starship exploded (E16-E18) and Musk mentioned a leak (E19); I need to verify if airlines were affected and resumed operations as claimed.

- **codex** `C|S_text_text` groundlie360_v2276004_r1g1_1 request_0009：`{"input_type": "text", "query": "Jimmy Kimmel Live Donald Trump April Fools campaign announcement March 31 2016", "output_type": "text"}` — E23 identifies a faux announcement on Jimmy Kimmel Live, while R4 was inaccessible; a targeted search can locate firsthand material or independent reporting about that broadcast.
- **qwen** `C|S_text_text` groundlie360_v1959500_r1_1 request_0006：`{"input_type": "text", "query": "Lipton Lemon Green tea worms video 2016", "output_type": "text"}` — Search for external reports about the 2016 claim of worms in Lipton Lemon Green tea to verify the video's allegations.

- **codex** `C|S_text_text` fmnv_c_1597049539495690244_r1g1_1 request_0006：`{"input_type": "text", "query": "CBS Weekend News \"COVID lockdown protests erupt in China\"", "output_type": "text"}` — E15 and E19 display this headline, which contradicts the title’s United States framing; locating the broadcaster’s report can establish the event’s context.
- **qwen** `C|S_text_text` fmnv_c_jdZthfG9NwA_r1_1 request_0006：`{"input_type": "text", "query": "Taraji P. Henson BET Awards 2024 speech Project 2025 voting", "output_type": "text"}` — Search for the original BET Awards 2024 speech to verify if Taraji P. Henson mentioned Project 2025 and voting, as the claim suggests.

- **codex** `C|S_text_text` fmnv_c_dKMlKAJiTrc_r1g1_1 request_0008：`{"input_type": "text", "query": "LAFD \"Silverman\" \"January\" \"2025\" \"return\"", "output_type": "text"}` — E19 reports the warning, but the remaining unread pages are an aggregator or concern other topics; a targeted search may find the original LAFD statement or independent reporting.
- **qwen** `C|S_text_text` fmnv_c_vumukVWqdZ4_r1_1 request_0006：`{"input_type": "text", "query": "2024 presidential election results Donald Trump Kamala Harris Associated Press", "output_type": "text"}` — Need external verification of 2024 election outcome and candidates to fact-check the claim against video evidence E4-E10 showing Trump elected.

- **codex** `C|S_text_text` groundlie360_v34049104_r1g1_1 request_0009：`{"input_type": "text", "query": "\"Houston\" \"Kroger\" \"plastic bag\" \"gasoline\" \"December 2019\"", "output_type": "text"}` — E16 reports the event and E17 dates that report to December 2019; a targeted search can seek independent contemporary evidence beyond the retrospective summary in E15.
- **qwen** `C|S_text_text` fmnv_c_i1rl1KhKIjE_r1_1 request_0006：`{"input_type": "text", "query": "Rachael Gunn Paris 2024 Olympic breaking performance audience reaction", "output_type": "text"}` — Search for external evidence about Rachael Gunn's Olympic breaking performance reception to verify the claim.

- **codex** `C|S_text_text` groundlie360_v26052400_r1g1_1 request_0006：`{"input_type": "text", "query": "Lincoln Tunnel water flooding July 14 2020 Port Authority Hudson River", "output_type": "text"}` — E14 shows water spilling beside a tunnel roadway, but does not establish its source, location or date; contemporaneous event reporting can test those parts of the claim.
- **qwen** `C|S_text_text` fmnv_c_ySOXzZQnQ_Q_r1_1 request_0006：`{"input_type": "text", "query": "Donald Trump wins 2024 presidential election CNBC Squawk Box November 2024", "output_type": "text"}` — Frame E16 shows 'PRESIDENT ELECT DONALD TRUMP' on CNBC; searching for the event confirms the election outcome and context.

- **codex** `C|S_text_text` fmnv_c_1524680668705402881_r1g1_1 request_0006：`{"input_type": "text", "query": "Metropolitan Police Partygate \"more than 100\" \"referrals\" \"fixed penalty notices\"", "output_type": "text"}` — E14–E16 display a caption reporting more than 100 additional referrals, contradicting the title; searching those terms targets external reporting of the announcement.
- **qwen** `C|S_text_text` groundlie360_v1958010_r1_1 request_0005：`{"input_type": "text", "query": "Kasich \"women came out of the kitchen\" speech February 2016", "output_type": "text"}` — Search for the event and exact quote to verify the claim's context and accuracy.

- **codex** `C|S_text_text` fmnv_c_1714273821681995805_r1g1_1 request_0009：`{"input_type": "text", "query": "UK Covid inquiry second phase October 2023 Module 2 political decisions lockdown public health scope hearings", "output_type": ` — Official pages R1 and R4 were not retrieved, and E19 alone does not establish every requirement; a date-specific search can find direct reporting of the second phase's scope.
- **qwen** `C|S_text_text` fmnv_c_doJ9iSPBUbg_r1_1 request_0006：`{"input_type": "text", "query": "57 million early votes 2024 US election", "output_type": "text"}` — The video (E1) claims 57 million early votes (E18); I must verify this figure with external sources to fact-check the claim.

- **codex** `C|S_text_text` fmnv_c_gPnKb8NEu5M_r1g1_1 request_0006：`{"input_type": "text", "query": "Madison Dane County four voters lawsuit uncounted absentee ballots 2024 presidential election", "output_type": "text"}` — E16 reports four voters suing Madison over uncounted ballots; external reporting or court records are needed to verify the lawsuit, election, and whether Dane County was also named.
- **qwen** `C|S_text_text` fmnv_c_RXR0B1SxiU0_r1_1 request_0006：`{"input_type": "text", "query": "Elon Musk Tesla Cybertruck Los Angeles fire department withdrawal", "output_type": "text"}` — The video frames (E15-E18) show Cybertrucks at a fire response site, but I need external evidence to verify the claim about Musk withdrawing them.

- **codex** `C|S_text_text` fmnv_c_1585081341191782401_r1g1_1 request_0008：`{"input_type": "text", "query": "\"Murthy\" \"Dickerson\" \"RSV\" \"2022\"", "output_type": "text"}` — The available pages concern other events or general respiratory illness; searching the named participants with RSV and 2022 may locate the interview itself.
- **qwen** `C|S_text_text` groundlie360_c_46581202_r1_1 request_0009：`{"input_type": "text", "query": "Bruce McCandless II first untethered spacewalk date STS-41B February 1984", "output_type": "text"}` — Targets the exact date of the first untethered spacewalk to verify the claim's timeline.

- **codex** `C|S_text_text` fmnv_c_3W1DFf3IEFw_r1g1_1 request_0006：`{"input_type": "text", "query": "Gavin Newsom confronted mother Rachel Los Angeles wildfires Sky News", "output_type": "text"}` — E16 shows a Sky News confrontation report, and E17 names Rachel; searching these clues can locate reporting that establishes the encounter’s context and whether the woman is a mother.
- **qwen** `C|S_text_text` fmnv_c_FEbcNwkD7EU_r1_1 request_0006：`{"input_type": "text", "query": "SpaceX Polaris Dawn crew spacewalk EVA Associated Press", "output_type": "text"}` — The video frames E15 and E16 show astronauts performing EVA activities with Associated Press branding; searching for official reports will verify the claim.

- **codex** `C|S_text_text` fmnv_c_aoq46QuPuMk_r1g1_1 request_0006：`{"input_type": "text", "query": "SpaceX Starship explodes debris night sky Bahamas Miami Guardian", "output_type": "text"}` — E12 identifies Guardian branding and Bahamas and Miami captions; a broader event search removes the exact-location restriction that returned no results.
- **qwen** `C|S_text_text` groundlie360_v183204_r1_1 request_0006：`{"input_type": "text", "query": "\"48 Hour Challenge\" game Facebook missing kids 2017", "output_type": "text"}` — The video (E1) discusses a '48 Hour Challenge' on Facebook; I will search for external reports about this claim from 2017 to verify its existence and context.

- **codex** `C|S_text_text` fmnv_c_E_m8oSqtRZM_r1g1_1 request_0006：`{"input_type": "text", "query": "SpaceX Starship loses control breaks apart Texas FAA Florida airports Kayla Gaskins National News Desk", "output_type": "text"}` — E16 identifies the presenter and news program, while E19 shows the Starship explosion headline and Florida airport disruption caption, providing specific clues to locate reporting on the event.
- **qwen** `C|S_text_text` fmnv_c_zcEuCznPmYM_r1_1 request_0006：`{"input_type": "text", "query": "Paris 2024 Olympics sports eliminated removed breaking debut new sport", "output_type": "text"}` — Search for external evidence about which sports were eliminated from Paris 2024 Olympics and whether breaking was a new Olympic sport to verify the claim's core assertions.

- **codex** `C|S_text_text` groundlie360_v23670804_r1g1_1 request_0005：`{"input_type": "text", "query": "Natalia Bryant February 2020 Kobe daughter death suicide BBC News", "output_type": "text"}` — E13 displays a memorial claim and BBC NEWS branding; external reporting is needed to verify the alleged death and its attribution.
- **qwen** `C|S_text_text` groundlie360_v1815915_r1_1 request_0006：`{"input_type": "text", "query": "2015-08-04)", "output_type": "text"}` — The video E1 shows AJ+ branding and discusses deported veterans, but the claim about a 'Bunker' in Mexico needs external verification. Searching for the 2015-08-04 posting date will help find when this content was publis

- **codex** `C|S_text_text` fmnv_c_E_m8oSqtRZM_r1g1_1 request_0009：`{"input_type": "text", "query": "SpaceX Starship eighth flight March 6 2025 loss attitude control ascent engines rapid unscheduled disassembly site:spacex.com",` — E22 identifies the eighth flight and quotes SpaceX; a targeted search for SpaceX’s account can establish the event date and directly support the reported loss of control and breakup.
- **qwen** `C|S_text_text` fmnv_c__0oux8o9MJo_r1_1 request_0006：`{"input_type": "text", "query": "Donald Trump wins 2024 presidential election", "output_type": "text"}` — Search for external evidence to verify the claim in Material A (E1) that Trump won the 2024 election.

- **codex** `C|S_text_text` fmnv_c_1615114422162276356_r1g1_1 request_0006：`{"input_type": "text", "query": "Baijiazhuang Gaolou Hebei January 2023 Covid cases restrictions Chen Wenda", "output_type": "text"}` — E14 identifies the report’s stated place and date, E15 names an interviewee, and E16 says cases surged after restrictions were dropped; these clues target reporting on the depicted event.
- **qwen** `C|S_text_text` fmnv_c_v56J3P8QvT0_r1_1 request_0006：`{"input_type": "text", "query": "Kalonzo Musyoka Rigathi Gachagua alliance 2027 elections", "output_type": "text"}` — The video frames show Gachagua in Machakos but do not contain Kalonzo or any statement about an alliance; external search is needed to verify the claim.

- **codex** `C|S_text_text` groundlie360_v34049104_r1g1_1 request_0010：`{"input_type": "text", "query": "\"woman\" \"gas\" \"plastic bag\" \"Kroger\" \"December 12, 2019\"", "output_type": "text"}` — E16 reports the relevant event, but the other results offer retrospective commentary or unrelated incidents; a targeted search for contemporary coverage can provide independent evidence predating the claim.
- **qwen** `C|S_text_text` groundlie360_v1921702_r1_1 request_0006：`{"input_type": "text", "query": "Ato Essandoh Django Unchained filming death 2013", "output_type": "text"}` — The claim states Ato Essandoh was killed during filming; I must search for external evidence about this event to verify it.

- **codex** `C|S_text_text` fmnv_c_1714273821681995805_r1g1_1 request_0006：`{"input_type": "text", "query": "UK Covid-19 Inquiry Module 2 scope early political decisions public health lockdown", "output_type": "text"}` — E14 and E15 show a COVID Inquiry hearing caption about lockdown decisions; an official Module 2 scope source can establish whether the claimed second-phase focus is accurate.
- **qwen** `C|S_text_text` groundlie360_v1936701_r1_1 request_0006：`{"input_type": "text", "query": "Bill Maher Real Time FCC fine joint smoke October 2010", "output_type": "text"}` — The video frames show a normal panel discussion with no visible smoking; external search is needed to verify the claim about an FCC fine.

- **codex** `C|S_text_text` groundlie360_v21694609_r1g1_1 request_0007：`{"input_type": "text", "query": "\"Rogers\" \"House Democrats Putting Illegal Immigrants In Front of Service Members\" \"2019\"", "output_type": "text"}` — E15’s title identifies a potentially relevant committee statement; this search seeks that event-specific source after the previous search failed to save.
- **qwen** `C|S_text_text` groundlie360_c_46819210_r1_1 request_0002：`{"input_type": "text", "query": "Louis Vuitton San Francisco sign thieves steal less than $950", "output_type": "text"}` — Initiating search for the specific claim about a Louis Vuitton sign in San Francisco regarding the $950 theft threshold.

- **codex** `C|S_text_text` fmnv_c_GJmlx8LwHug_r1g1_1 request_0008：`{"input_type": "text", "query": "site.gov.ca.gov January 11 2025 Newsom Mexico firefighters arrival thanks", "output_type": "text"}` — E20 reports Newsom’s gratitude and the firefighters’ arrival; an official California statement would provide first-hand corroboration of the event.
- **qwen** `C|S_text_text` fmnv_c_5ah45BH3ZHU_r1_1 request_0006：`{"input_type": "text", "query": "Eliud Kipchoge Paris Olympics 2024 marathon winner results", "output_type": "text"}` — The video frames show a runner in a rural setting with police escort, not the Paris Olympics marathon; I need to verify who actually won the Paris 2024 Olympic marathon and whether Kipchoge competed.

### page
- **codex** `C|R_page` fmnv_c_ENia1MRCIEI_r1g1_1 request_0010：`{"record_id": "E28"}` — E28 is SpaceX’s own statement about a long-duration static fire ahead of the eighth flight test, making it the most direct source to check the claimed event and any stated duration.
- **qwen** `C|R_page` fmnv_c_S9uZ7tPnY80_r1_1 request_0007：`{"record_id": "E24"}` — Reading ABC7 news page (E24) to verify timeline and descriptions of post-wildfire landscape conditions mentioned in search results.

- **codex** `C|R_page` fmnv_c_7HN8Apdk28c_r1g1_1 request_0008：`{"record_id": "E24"}` — E24 reports the same flight and may provide corroboration beyond E22; E23 is another copy of the ABC News report.
- **qwen** `C|R_page` fmnv_c_ue4pMOsllxc_r1_1 request_0009：`{"record_id": "E24"}` — E24 contains the Teamsters polling data referenced in the video's OCR (E12); needed to verify the polling claims shown in Material A.

- **codex** `C|R_page` fmnv_c_DNkJSeY1jRg_r1g1_1 request_0008：`{"record_id": "E16"}` — E16 reports the meeting itself and can corroborate the event date and post-election context alongside the official remarks in E15.
- **qwen** `C|R_page` fmnv_c_RXR0B1SxiU0_r1_1 request_0009：`{"record_id": "E21"}` — Reading E21 (Yahoo Finance) to verify the claim against another mainstream source before concluding on the withdrawal vs. diversion discrepancy.

- **codex** `C|R_page` fmnv_c_1524680668705402881_r1g1_1 request_0009：`{"record_id": "E21"}` — E21 reports the relevant police announcement and may provide its exact wording and date, allowing the claimed absence of referrals to be checked against contemporaneous reporting.
- **qwen** `C|R_page` fmnv_c__F3RKk9srG4_r1_1 request_0007：`{"record_id": "E21"}` — E21's title directly references the 'main character' claim; reading will show what evidence supports the viral status assertion.

- **codex** `C|R_page` groundlie360_v21374806_r1g1_1 request_0007：`{"record_id": "E24"}` — E24 reports the claimed Florida cocaine discovery and quotes police, making it relevant for checking what was recovered and whether Hurricane Dorian’s causal role is supported.
- **qwen** `C|R_page` fmnv_c_vumukVWqdZ4_r1_1 request_0010：`{"record_id": "E16"}` — Read E16 electoral college results to confirm Trump's victory details before finalizing the fact-check report on the claim about Biden.

- **codex** `C|R_page` groundlie360_v27961928_r1g1_1 request_0008：`{"record_id": "E14"}` — E14 reports the Biden bus incident directly and may provide attribution or details relevant to the claimed departure along 11th Street toward I-35.
- **qwen** `C|R_page` fmnv_c_jdZthfG9NwA_r1_1 request_0010：`{"record_id": "E21"}` — Reading E21 to verify the specific quote and context of Taraji P. Henson's speech regarding Project 2025 at the BET Awards.

- **codex** `C|R_page` groundlie360_v46441308_r1g1_1 request_0007：`{"record_id": "E16"}` — E16 reports the event directly and can provide independent corroboration of E15, including the incident date and what happened on the stairs.
- **qwen** `C|R_page` groundlie360_v183204_r1_1 request_0008：`{"record_id": "E21"}` — Reading E21 (Metro) to gather details on the reported incident and Facebook's response, complementing the hoax confirmation in E22.

- **codex** `C|R_page` fmnv_c_E_m8oSqtRZM_r1g1_1 request_0010：`{"record_id": "E23"}` — E23 is a news outlet’s report about Starship explosions and aviation impacts that may clarify the event and provide corroboration after the official pages failed to load.
- **qwen** `C|R_page` groundlie360_v183204_r1_1 request_0009：`{"record_id": "E24"}` — Reading E24 (FOX 11) to verify its statement that the challenge appears false, complementing the hoax confirmation in E22.

- **codex** `C|R_page_more` groundlie360_v2256709_r1g1_1 request_0010：`{"record_id": "E15", "offset": 6000}` — E15 continues beyond offset 6000; its remaining text may detail the subsequent clarification needed to distinguish the quoted interview remark from Trump's later stated position.
- **qwen** `C|R_page_more` fmnv_c_UkbuN8grJmY_r1_1 request_0009：`{"record_id": "E23", "offset": 6000}` — Continuing to read CNN article E23 to capture full details on debris reports and video attribution.

- **codex** `C|R_page_more` groundlie360_v21694609_r1g1_1 request_0010：`{"record_id": "E16", "offset": 6000}` — Continue E16 at offset 6000 to seek the proposed amendment’s actual terms and vote outcome, which are needed to assess whether the title accurately describes the event.
- **qwen** `C|R_page_more` fmnv_c_XGC31lmdS6s_r1_1 request_0008：`{"record_id": "E17", "offset": 6000}` — Continue reading E17 to get full article content about who recovered the Super Heavy booster.

- **codex** `C|R_page_more` groundlie360_v46443403_r1g1_1 request_0008：`{"record_id": "E19", "offset": 6000}` — E19 reports the relevant press conference directly and has unread text that may establish the context of the quoted warning.
- **qwen** `C|R_page_more` groundlie360_v1921702_r1_1 request_0010：`{"record_id": "E23", "offset": 6000}` — Reading the continuation of E23 to confirm Ato Essandoh's post-2013 career and explicit denial of the death rumor.

- **codex** `C|S_dates` fmnv_c_1701376058795139538_r1g1_1 request_0009：`{"source": "R1", "output_type": "text", "include": ["dates"]}` — E22 directly documents approval and emergency authorization of the updated 2023–2024 vaccines; dating R1 will establish the temporal scope of this evidence against the rejection claim.
- **qwen** `C|S_dates` groundlie360_v1936701_r1_1 request_0010：`{"source": "R6", "output_type": "text", "include": ["dates"]}` — Checking the publication date of the CBS News article (R6/E23) which explicitly states the joint was fake, to establish a primary source timeline against the claim.

- **codex** `C|S_dates` groundlie360_v38706501_r1g1_1 request_0010：`{"source": "R7", "output_type": "text", "include": ["dates"]}` — E19 contains McDonald’s first-hand announcement of charging bicycles in its Jieyang restaurant; checking R7’s publication and archive dates would strengthen whether this evidence predates the December 16, 2021 claim.
- **qwen** `C|S_dates` groundlie360_c_46581202_r1_1 request_0006：`{"source": "R6", "output_type": "text", "include": ["dates"]}` — Check publication date of R6 (NASA STS-41B mission page) to verify source timing relative to the 2023 claim.

- **codex** `C|S_dates` fmnv_c_1503424514125934593_r1g1_1 request_0009：`{"source": "R1", "output_type": "text", "include": ["dates"]}` — R1 directly reports the removal of UK travel restrictions; checking its publication date will help anchor “this week” to the historical announcement rather than imply a current event.
- **qwen** `C|S_dates` fmnv_c__F3RKk9srG4_r1_1 request_0010：`{"source": "R7", "output_type": "text", "include": ["dates"]}` — Need publication date for R7 (E22) to establish timeline of viral coverage about Kim Yeji alongside R4's July 31 date (E23).

- **codex** `C|S_dates` groundlie360_v35136101_r1g1_1 request_0009：`{"source": "R2", "output_type": "text", "include": ["dates"]}` — Checking R2’s publication date can establish whether its reporting on the interview predates the June 20, 2021 claim.
- **qwen** `C|S_dates` groundlie360_v1988600_r1_1 request_0009：`{"source": "R4", "output_type": "text", "include": ["dates"]}` — Checking the publication date of R4 (Knox Road, May 2009) to verify if it predates or matches the claim's 2009-01-20 posting date.

- **codex** `C|S_dates` fmnv_c_dGMu_MEPY70_r1g1_1 request_0010：`{"source": "R10", "output_type": "text", "include": ["dates"]}` — Dating R10’s official account of inspections in both fire areas helps establish the timeline relevant to the title’s claim that the damage assessment maps were new.
- **qwen** `C|S_dates` fmnv_c__F3RKk9srG4_r1_1 request_0009：`{"source": "R4", "output_type": "text", "include": ["dates"]}` — Need to verify publication dates of sources E21 (R4) and E22 (R7) to establish timeline of the viral claim about Kim Yeji.

- **codex** `C|S_dates` groundlie360_v17396202_r1g1_1 request_0008：`{"source": "R6", "output_type": "text", "include": ["dates"]}` — E25 directly supports the reported Southern Ocean filming and camera provenance; dating R6 will clarify its timing relative to the claim’s stated 2018-10-20 posting.
- **qwen** `C|S_dates` fmnv_c_X9tDERFjOMw_r1_1 request_0009：`{"source": "R7", "output_type": "text", "include": ["dates"]}` — Checking publication date of BBC article (R7/E22) to establish timeline relative to the claim and confirm the successful booster catch was reported contemporaneously.

- **codex** `C|S_dates` groundlie360_v34049104_r1g1_1 request_0008：`{"source": "R4", "output_type": "text", "include": ["dates"]}` — E16 reports the event itself; checking R4’s publication and archive dates can establish whether this account predates May 2021, though correspondence with the given video still needs verification.
- **qwen** `C|S_dates` groundlie360_v1914403_r1_1 request_0010：`{"source": "R4", "output_type": "text", "include": ["dates"]}` — Check publication date of Business Insider article (R4/E16) to verify if it predates the claim's 2015-11-30 posting.

- **codex** `C|S_dates` groundlie360_v26016501_r1g1_1 request_0008：`{"source": "R6", "output_type": "text", "include": ["dates"]}` — Check whether R6 predates the claimed January 2020 post; its publication date can constrain chronology but cannot establish the video's origin or validate the caption.
- **qwen** `C|S_dates` groundlie360_c_46819210_r1_1 request_0004：`{"source": "R2", "output_type": "text", "include": ["dates"]}` — Checking the publication date of R2 (Patch) to verify if the report predates or aligns with the claim date of 2024-07-01.

- **codex** `C|S_open_video` groundlie360_v2276004_r1g1_1 request_0008：`{"source": "https://www.youtube.com/watch?v=VO-1ePYypdU", "output_type": "video"}` — E23 describes a fabricated Jimmy Kimmel Live announcement assembled from old Trump footage and voice-over, and links this YouTube video. Opening it can obtain a candidate original for comparison with Material A, whose co
- **qwen** `C|S_open_video` groundlie360_v1958010_r1_1 request_0007：`{"source": "https://www.youtube.com/watch?v=GrVOm-EVcY0", "output_type": "video"}` — The BBC article (E16) cites a YouTube video (GrVOm-EVcY0) as the source of Kasich's 'kitchens' remark and notes early clips lacked context. Opening this video will allow direct verification of the full speech to confirm 

### action
- **codex** `B|I_asr` videosham_4197_r1g1_1 request_0006：`{"targets": ["E2"], "request": {"operation": "asr", "params": {}}}` — E25 reports matching audio windows, but E32 covers speech recognition only for Material A. Running ASR on Material B (E2) will check for recoverable speech across its 8.575 seconds, allowing comparison without treating m
- **qwen** `B|I_asr` videosham_1028_v9_r1_1 request_0007：`{"targets": ["E1"], "request": {"operation": "asr", "params": {}}}` — OCR shows E25 (Material B) has text "A conversation about CRPS" while E28 (Material A) has "lorem ipsum" at the same timestamp 0.0s, indicating a visual difference; running ASR on Material A (E1) will check whether the a

- **codex** `B|I_asr` videosham_4224_r1g1_1 request_0005：`{"targets": ["E1"], "request": {"operation": "asr", "params": {}}}` — E5 and E14 show a visual difference near the goal at 0.500 s, but the supplied observations do not establish audio content. ASR on Material A (E1) will obtain timestamped speech and sound observations needed to assess it
- **qwen** `B|I_asr` videosham_4005_r1_1 request_0007：`{"targets": ["E1"], "request": {"operation": "asr", "params": {}}}` — The compare_media result E25 confirms the audio tracks match, but the task requires stating how differences affect meaning. Running ASR on Material A (E1) will obtain the spoken transcript to document the audio content a

- **codex** `B|I_compare_video` videosham_4008_r1g1_1 request_0002：`{"targets": ["E1", "E2"], "request": {"operation": "compare_media", "params": {}}}` — E4 and E9 show “VIENAS DU TRYS” on the foreground book, while E15 and E20 show “THE TRINITARIAN FAITH.” Comparing E1 with E2 will establish temporal correspondence and identify visual and audio differences across both cl
- **qwen** `B|I_compare_video` vcsl_pair_8_r1_1 request_0002：`{"targets": ["E1", "E2"], "request": {"operation": "compare_media", "params": {}}}` — Only four isolated frames from E1 and E2 are visible, insufficient to establish correspondences across the full 115-second videos; compare_media will align both videos end-to-end to identify identical or transformed segm

- **codex** `B|I_compare_video` videosham_2004_r1g1_1 request_0002：`{"targets": ["E1", "E2"], "request": {"operation": "compare_media", "params": {}}}` — E4 and E9 from Material A and E15 and E20 from Material B show a drummer in the same setting, but these isolated frames do not establish temporal correspondence or audiovisual differences. Comparing E1 with E2 will obtai
- **qwen** `B|I_compare_video` vcsl_pair_9_r1_1 request_0002：`{"targets": ["E1", "E2"], "request": {"operation": "compare_media", "params": {}}}` — The task requires finding corresponding segments between videos E1 and E2 with time ranges. Only four isolated frames are currently visible, which is insufficient to establish continuous correspondences or transformation

- **codex** `B|I_frames_box` videosham_4008_r1g1_1 request_0004：`{"targets": ["E1"], "request": {"operation": "get_frames", "params": {"times_s": [0.0, 0.7, 1.5], "box": [0.395, 0.365, 0.755, 0.705]}}}` — E21 shows different book-cover wording in A and B, but does not establish its extent over time. Extracting corresponding crops from A while retaining B’s E26–E28 will check the cover at 0.00, 0.68 and 1.48 seconds, inclu
- **qwen** `B|I_frames_box` videosham_4059_r1_1 request_0006：`{"targets": ["E1"], "request": {"operation": "get_frames", "params": {"times_s": [0.0, 2.8, 5.45], "box": [0.455, 0.465, 0.595, 0.665]}}}` — E22 shows a difference region in B at x 0.48-0.57 y 0.49-0.64 (text banner area). Cropped frames E34-E36 from B at 0.0, 2.8 and 5.433 s show this region; extracting the same box from A at the same times will allow direct

- **codex** `B|I_frames_times` vcsl_pair_g1_02_r1g1_1 request_0010：`{"targets": ["E1"], "request": {"operation": "get_frames", "params": {"times_s": [77.747, 80.747, 81.198]}}}` — E32 and E30 show black frames, which cannot establish a time offset. Sampling A at 77.747 and 81.198 s alongside 80.747 s checks additional points in the tail flagged by E34 for distinguishable content, without establish
- **qwen** `B|I_frames_times` vcsl_pair_9_r1_1 request_0005：`{"targets": ["E1"], "request": {"operation": "get_frames", "params": {"times_s": [0.3, 80.748]}}}` — compare_media identified A 77.2-81.2 s as having no counterpart in B; extracting frames from A near its start (0.3 s) and end (80.748 s) will show what this unique segment contains and whether it fits the surrounding foo

- **codex** `B|I_locate` vcsl_pair_g1_02_r1g1_1 request_0012：`{"targets": ["E2"], "request": {"operation": "locate", "params": {"query": {"type": "image", "evidence_id": "E36"}, "range_s": [67.494, 81.238], "top_k": 3}}}` — E32 at A 80.747 s and E30 at B 77.494 s show uninformative black frames, so they cannot establish an offset. Locating candidates for A’s later frame E36 within B 67.494–81.238 s tests for a distinguishable endpoint count
- **qwen** `B|I_locate` qvhighlights_603_r1_1 request_0002：`{"targets": ["E1"], "request": {"operation": "locate", "params": {"query": {"type": "text", "text": "Finger points to different areas of the water"}, "top_k": 6` — The four sampled frames (E3, E5, E8, E10) show a man gesturing near water but do not clearly depict a finger pointing to specific areas. To locate the exact segment matching the task description, I will use INSPECT with 

- **codex** `B|I_ocr_image` videosham_4025_r1g1_1 request_0005：`{"targets": ["E29"], "request": {"operation": "ocr", "params": {}}}` — E29 and E26 show different sweater colors at 0.000 s, but no clearly readable text. OCR on E29 will check for recoverable lettering in that crop; it cannot establish the cause of the color difference or confirm B’s sourc
- **qwen** `B|I_ocr_image` videosham_1028_v9_r1_1 request_0006：`{"targets": ["E25"], "request": {"operation": "ocr", "params": {}}}` — OCR of E28 (Material A) returned 'lorem ipsum', while the attached image E25 (Material B, same time/region) clearly shows different text ('A conversation about CRPS'). Running OCR on E25 will obtain the exact transcripti

- **codex** `C|I_asr` fmnv_c_eLD1XA2UcwQ_r1g1_1 request_0002：`{"targets": ["E1"], "request": {"operation": "asr", "params": {}}}` — E3, E5, E8 and E10 show a bulldozer among cars in smoky conditions, but do not establish residents evacuating or what is being cleared. ASR on E1 will obtain any intelligible speech and its timing, potentially providing 
- **qwen** `C|I_asr` fmnv_c_TIu19O02X2c_r1_1 request_0002：`{"targets": ["E1"], "request": {"operation": "asr", "params": {}}}` — The claim states Starship 'successfully completes flight test' and airlines 'resume normal operations', but frame E5 shows an explosion with caption 'explodes in flight test'. ASR on video E1 will capture the full narrat

- **codex** `C|I_compare_video` fmnv_c_fO5oQqZiXxw_r1g1_1 request_0010：`{"targets": ["E1", "E24"], "request": {"operation": "compare_media", "params": {}}}` — E26 documents Beyoncé’s Team USA introduction, while R7’s preliminary matching found no sampled picture correspondence with Material A. Comparing E1 with E24 will test visual and audio correspondence across both videos, 
- **qwen** `C|I_compare_video` groundlie360_v1958010_r1_1 request_0009：`{"targets": ["E1", "E17"], "request": {"operation": "compare_media", "params": {}}}` — SEARCH confirmed R3 (E17) contains 100% of Material A's visuals; running compare_media on [E1, E17] will precisely align the audio and video to verify if the claim title matches the actual speech content and context foun

- **codex** `C|I_frames_times` fmnv_c_1701376058795139538_r1g1_1 request_0005：`{"targets": ["E1"], "request": {"operation": "get_frames", "params": {"times_s": [3.22, 7.72, 10.36, 15.16, 17.66, 19.24]}}}` — E13 shows approval wording that conflicts with the rejection claim, but its montage limits detail. Frames from E1 at 3.22–19.24 seconds will provide clearer views of the headline and manufacturing and syringe scenes; the
- **qwen** `C|I_frames_times` groundlie360_v183204_r1_1 request_0005：`{"targets": ["E1"], "request": {"operation": "get_frames", "params": {"times_s": [3.08, 8.54, 11.72, 18.29, 27.0, 35.45]}}}` — E13 shows 12 shots in E1, but only four frames (E3/E5/E8/E10) have been viewed; extracting additional frames at the specified times will cover more shots and unconfirmed ranges to verify the video's content against the c

- **codex** `C|I_ocr_video` groundlie360_v26080509_r1g1_1 request_0003：`{"targets": ["E1"], "request": {"operation": "ocr", "params": {}}}` — E11 transcribes an obligation to care for undocumented people, but the video's source and the title's taxpayer-funding interpretation remain unverified. OCR on E1 will check visible text, including the FOX 11 microphone 
- **qwen** `C|I_ocr_video` groundlie360_v1815915_r1_1 request_0003：`{"targets": ["E1"], "request": {"operation": "ocr", "params": {}}}` — OCR on E1 will extract any on-screen text, titles or captions not captured by ASR (E11), complementing the visible banner in E8 and ensuring all textual claims in Material A are recorded for verification.

- **codex** `C|I_video_structure` groundlie360_v28763910_r1g1_1 request_0004：`{"targets": ["E1"], "request": {"operation": "video_structure", "params": {}}}` — E3, E5 and E8 show the opening title, milk wash and frying, but leave intervening footage unobserved. video_structure will map E1’s shots and any frozen, repeated or black spans to identify intervals needing inspection; 
- **qwen** `C|I_video_structure` fmnv_c_XGC31lmdS6s_r1_1 request_0004：`{"targets": ["E1"], "request": {"operation": "video_structure", "params": {}}}` — Frames E3-E10 show a booster ascent and landing sequence, but OCR (E12) found no text and ASR (E11) detected no speech. Running video_structure on E1 will identify shots, repeats, and frozen spans to check for edits befo

- **codex** `C|R_frames_review` fmnv_c_1714273821681995805_r1g1_1 request_0011：`{"record_id": "E11"}` — E21 confirms that Module 2 examined political decision-making, including the initial response. Reading E11 will reveal the video's transcribed speech and its reliability limits, while the selected frames provide visual c
- **qwen** `C|R_frames_review` fmnv_c_ySOXzZQnQ_Q_r1_1 request_0011：`{"record_id": "E11"}` — External sources (E21-E24) confirm Donald Trump won the 2024 election, but the specific content of Material A (E1) must be verified to fact-check its title claim. Reading the ASR transcript (E11) is necessary to determin

- **codex** `C|R_page` fmnv_c_ENia1MRCIEI_r1g1_1 request_0010：`{"record_id": "E28"}` — E28 is SpaceX’s own statement about a long-duration static fire ahead of the eighth flight test, making it the most direct source to check the claimed event and any stated duration.
- **qwen** `C|R_page` fmnv_c_S9uZ7tPnY80_r1_1 request_0007：`{"record_id": "E24"}` — Reading ABC7 news page (E24) to verify timeline and descriptions of post-wildfire landscape conditions mentioned in search results.

- **codex** `C|R_page_more` groundlie360_v2256709_r1g1_1 request_0010：`{"record_id": "E15", "offset": 6000}` — E15 continues beyond offset 6000; its remaining text may detail the subsequent clarification needed to distinguish the quoted interview remark from Trump's later stated position.
- **qwen** `C|R_page_more` fmnv_c_UkbuN8grJmY_r1_1 request_0009：`{"record_id": "E23", "offset": 6000}` — Continuing to read CNN article E23 to capture full details on debris reports and video attribution.

- **codex** `C|R_record` groundlie360_v32202105_r1g1_1 request_0016：`{"record_id": "E19"}` — E14 reports the March 19, 2021 Georgia departure, but its connection to E1 remains unverified. Reading E19 will obtain the Washington Post candidate’s original description and any footage context useful for establishing 
- **qwen** `C|R_record` fmnv_c_X9tDERFjOMw_r1_1 request_0008：`{"record_id": "E21"}` — E21 contains the Mechazilla catch record showing booster catch outcomes, directly relevant to verifying whether the booster was lost or caught as claimed.

- **codex** `C|S_dates` fmnv_c_1701376058795139538_r1g1_1 request_0009：`{"source": "R1", "output_type": "text", "include": ["dates"]}` — E22 directly documents approval and emergency authorization of the updated 2023–2024 vaccines; dating R1 will establish the temporal scope of this evidence against the rejection claim.
- **qwen** `C|S_dates` groundlie360_v1936701_r1_1 request_0010：`{"source": "R6", "output_type": "text", "include": ["dates"]}` — Checking the publication date of the CBS News article (R6/E23) which explicitly states the joint was fake, to establish a primary source timeline against the claim.

- **codex** `C|S_open_video` groundlie360_v2276004_r1g1_1 request_0008：`{"source": "https://www.youtube.com/watch?v=VO-1ePYypdU", "output_type": "video"}` — E23 describes a fabricated Jimmy Kimmel Live announcement assembled from old Trump footage and voice-over, and links this YouTube video. Opening it can obtain a candidate original for comparison with Material A, whose co
- **qwen** `C|S_open_video` groundlie360_v1958010_r1_1 request_0007：`{"source": "https://www.youtube.com/watch?v=GrVOm-EVcY0", "output_type": "video"}` — The BBC article (E16) cites a YouTube video (GrVOm-EVcY0) as the source of Kasich's 'kitchens' remark and notes early clips lacked context. Opening this video will allow direct verification of the full speech to confirm 

- **codex** `C|S_text_text` groundlie360_v46443403_r1g1_1 request_0010：`{"input_type": "text", "query": "site.president.gov.ua 24 February 2023 press conference America NATO Baltic states sons daughters", "output_type": "text"}` — R4–R8 could not be opened, so a targeted search for the presidential account may establish the original context of the disputed statement.
- **qwen** `C|S_text_text` fmnv_c_TIu19O02X2c_r1_1 request_0006：`{"input_type": "text", "query": "SpaceX Starship flight test January 16 2025 explosion airlines resume operations", "output_type": "text"}` — The video shows Starship exploded (E16-E18) and Musk mentioned a leak (E19); I need to verify if airlines were affected and resumed operations as claimed.

