#!/usr/bin/env python3
"""
Synthetic Social Media Dataset Generator
Generates realistic social media post data for BCA internship project:
'Social Media Sentiment and Trend Analysis'

Dataset Schema:
- post_id: Unique identifier (e.g. POST_0001)
- platform: Instagram, X, Facebook, YouTube
- username: Fictional handle
- post_text: Realistic social media text
- hashtags: Space-separated hashtags
- likes: Non-negative integer
- comments: Non-negative integer
- shares: Non-negative integer
- post_date: ISO 8601 formatted timestamp (YYYY-MM-DD HH:MM:SS)
- topic: Category (Technology, Business, Health, Entertainment, Sports, Climate, Education, Lifestyle)
"""

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

# Set fixed seed for reproducibility while maintaining rich diversity
random.seed(42)

PLATFORMS = ["X", "Instagram", "Facebook", "YouTube"]
PLATFORM_WEIGHTS = [0.35, 0.30, 0.20, 0.15]

USERNAMES = [
    "@tech_insider", "@code_wizard", "@market_pulse", "@daily_health", "@film_fanatic",
    "@eco_future", "@sports_corner", "@ai_innovations", "@data_enthusiast", "@future_trends",
    "@cyber_sentinel", "@global_fin", "@healthy_living", "@stream_queen", "@gridiron_fan",
    "@green_planet", "@edutech_pro", "@biz_leader", "@gadget_guru", "@mindful_wellness",
    "@cinematics", "@stadium_talk", "@climate_action", "@python_dev", "@crypto_watcher",
    "@fit_life", "@indie_gamer", "@hoops_central", "@clean_energy", "@stem_educator",
    "@venture_scout", "@ux_designer", "@holistic_health", "@box_office_hub", "@esports_arena",
    "@conservationist", "@campus_life", "@cloud_architect", "@fintech_radar", "@nutrition_first",
    "@music_vibes", "@champions_league", "@solar_power", "@future_skills", "@startup_founder",
    "@mobile_dev", "@mental_peace", "@binge_watcher", "@athlete_journal", "@sustainability_now",
    "@code_craftsman", "@equity_analyst", "@yoga_everyday", "@critics_choice", "@soccer_mania",
    "@earth_patrol", "@academic_insights", "@devops_ninja", "@market_maven", "@wellness_journey"
]

TOPICS = [
    "Technology", "Business", "Health", "Entertainment",
    "Sports", "Climate", "Education", "Lifestyle"
]

# Post templates by Topic and Sentiment: (Text template, hashtags)
TEMPLATES = {
    "Technology": {
        "Positive": [
            ("The new AI breakthroughs in real-time language translation are breathtaking! Open source models are democratizing access for everyone. Loving the incredible progress.", "#AI #Innovation #FutureTech #MachineLearning"),
            ("Just upgraded to Python 3.12 and the speed improvement in standard library performance is awesome! Clean code and faster execution.", "#Python #Programming #DevLife #Coding"),
            ("Our team successfully shipped our cloud migration with zero downtime! Kudos to the DevOps engineers who made this seamless transition happen.", "#CloudComputing #DevOps #Engineering #AWS"),
            ("The new battery technology could double electric vehicle range while cutting charging time in half. Exciting times for green tech!", "#CleanTech #EV #ElectricVehicles #Innovation"),
            ("Open source community never ceases to amaze me. Collaborating with talented developers across continents is an absolute joy.", "#OpenSource #Collaboration #SoftwareDevelopment #TechCommunity"),
            ("Impressed by the latest cybersecurity defenses against zero-day exploits. Great work by threat intelligence teams everywhere.", "#CyberSecurity #InfoSec #TechNews #Privacy"),
            ("Generative AI coding assistants saved me hours of boilerplate writing today. More time to focus on complex software architecture!", "#AICoding #Productivity #SoftwareEngineer #Tech"),
            ("The camera sensors and image stabilization on this new device produce studio-quality photos. Absolutely phenomenal hardware design.", "#TechReview #MobilePhotography #Gadgets #Innovation"),
            ("Quantum computing research reached another milestone in error mitigation. Theoretical computer science turning into practical reality!", "#QuantumComputing #DeepTech #Science #Innovation"),
            ("Super proud to release our open source data analytics library today. Check out our GitHub repo and give us your feedback!", "#DataScience #Analytics #Python #GitHub")
        ],
        "Negative": [
            ("The latest operating system update completely broke my production workflow and wiped unsaved configurations. Shockingly poor quality control.", "#TechFail #Bugs #Frustrated #UpdateFail"),
            ("Major security vulnerability discovered in popular cloud framework. Zero communication from the vendor for over 48 hours. Totally unacceptable.", "#CyberAttack #SecurityBreach #CloudSecurity #TechDisaster"),
            ("Battery life on this flagship smartphone is atrocious. Barely lasts 3 hours of moderate screen time. Do not recommend buying this.", "#TechReview #BatteryDrain #BadHardware #Disappointed"),
            ("Cloud hosting costs have skyrocketed by 40% with zero performance improvement. Time to look for alternatives and migrate off this platform.", "#CloudCosts #SaaS #CloudComputing #Overpriced"),
            ("App crashes every single time I try to upload high-resolution media. Customer support has ignored my ticket for two weeks straight.", "#BugReport #AppCrash #CustomerFail #TechSupport"),
            ("Another social media data leak affecting millions of users. Tech giants must be held accountable for neglecting privacy rights.", "#DataPrivacy #DataLeak #CyberSecurity #TechEthics"),
            ("Terrible ergonomics and sluggish touchscreen responsiveness on this overpriced tablet. Complete waste of money.", "#BadDesign #HardwareFail #GadgetReview #Refund"),
            ("Forced subscription models for basic software features are getting out of hand. Developers deserve compensation, but this is pure gouging.", "#SubscriptionGreed #SoftwareDebate #ConsumerRights #TechIndustry"),
            ("The new developer API limits are crippling third-party tool builders. Disappointing move that harms the entire developer ecosystem.", "#APILimits #DevCommunity #TechNews #Disappointed"),
            ("Constant thermal throttling and fan noise under light workloads. Terrible cooling architecture in this expensive laptop.", "#HardwareIssues #LaptopReview #Overheating #TechFail")
        ],
        "Neutral": [
            ("Python steering council publishes PEP draft detailing proposed type system enhancements for upcoming releases.", "#Python #Programming #PEP #SoftwareDevelopment"),
            ("Tech conference keynote kicks off today at 10 AM PST. Live streams available across major video platforms.", "#TechConference #Keynote #LiveStream #TechNews"),
            ("Benchmark analysis comparing Rust and Go memory utilization in microservices architectures published by engineering blog.", "#Rust #GoLang #Microservices #SoftwareArchitecture"),
            ("Semiconductor manufacturers report annual production volume metrics for silicon wafer fabrication plants.", "#Semiconductors #Hardware #TechIndustry #Manufacturing"),
            ("Survey shows 64% of enterprise organizations are testing private large language model deployments on premise.", "#EnterpriseAI #Survey #DataAnalytics #TechTrends"),
            ("New standard specifications for next-generation USB connectivity standards announced by standards consortium.", "#HardwareStandards #USB #TechNews #Engineering"),
            ("Comparison of relational database indexing algorithms across high-concurrency read and write workloads.", "#Database #SQL #Performance #Backend"),
            ("Operating system vendor releases monthly scheduled security patch rollup addressing 18 reported CVEs.", "#SecurityPatch #InfoSec #SysAdmin #Updates"),
            ("Overview of modern front-end frameworks and their rendering pipelines for scalable single page web applications.", "#WebDev #Frontend #JavaScript #TechStack"),
            ("Data analytics report examines global smartphone shipment volumes across regional distribution markets.", "#MarketResearch #MobileTech #IndustryTrends #Data")
        ]
    },
    "Business": {
        "Positive": [
            ("Record quarterly revenue and 150% annual customer growth! Our team’s relentless dedication to product quality is paying off.", "#BusinessGrowth #StartupLife #Milestone #RevenueGrowth"),
            ("Excited to announce our successful Series B funding round to scale sustainable packaging solutions worldwide!", "#VentureCapital #StartupFunding #Sustainability #Entrepreneurship"),
            ("Honored to be recognized as one of the best workplaces for employee wellbeing and professional development.", "#CompanyCulture #GreatPlaceToWork #Leadership #HR"),
            ("Strategic partnership finalized today. Combining our analytics engine with their global distribution network creates immense market value.", "#Partnership #BusinessStrategy #Fintech #GlobalTrade"),
            ("Customer satisfaction score hit an all-time high of 98% this month. Huge shoutout to our customer success and support staff!", "#CustomerSuccess #CSAT #CustomerDelight #ClientAppreciation"),
            ("Small businesses are thriving in the digital economy through affordable cloud tools and direct-to-consumer channels.", "#SmallBiz #Economy #DigitalTransformation #Entrepreneur"),
            ("Stock markets rally as core inflation data cools faster than projected by financial analysts.", "#StockMarket #Investing #Economy #BullMarket"),
            ("Our green supply chain initiative reduced freight carbon emissions by 35% while cutting operational logistics costs.", "#Sustainability #SupplyChain #CostOptimization #GreenBusiness"),
            ("Launched our student internship and mentorship program today. Investing in the next generation of business talent!", "#Internship #CareerGrowth #Mentorship #NextGenTalent"),
            ("Surpassed one million active registered business users on our SaaS platform. Grateful for our vibrant community!", "#SaaS #Milestone #Community #TechStartup")
        ],
        "Negative": [
            ("Severe supply chain bottlenecks and container shipping rate spikes are hurting small manufacturing margins.", "#SupplyChainCrisis #Logistics #Inflation #Manufacturing"),
            ("Unexpected layoffs announced today across corporate divisions. Heartbreaking news for so many dedicated colleagues.", "#Layoffs #JobLoss #CorporateNews #TechLayoffs"),
            ("Interest rate hikes are strangling credit access for independent startups and early-stage ventures.", "#Financing #CreditCrunch #StartupStruggles #Economy"),
            ("Corporate greed exposed: record prices charged to consumers while supplier payments were cut by 20%.", "#PriceGouging #ConsumerAdvocacy #CorporateEthics #Economy"),
            ("Poor communication and micro-management continue to ruin company culture and drive away top talent.", "#ToxicWorkplace #BadLeadership #CompanyCulture #Resignation"),
            ("Commercial real estate vacancy rates hit unprecedented highs, posing risks to regional bank balance sheets.", "#RealEstate #Banking #EconomicRisks #CommercialProperty"),
            ("Retail sales plummet for the second straight month as consumer spending power erodes under sticky inflation.", "#RetailNews #ConsumerSpending #RecessionWarning #Economy"),
            ("Deceptive marketing practices uncovered in consumer subscription service. Regulators must step in.", "#ConsumerAlert #FraudWatch #BusinessEthics #Regulatory"),
            ("Supply shortage forces production line shutdown for third time this quarter. Massive financial losses recorded.", "#SupplyShortage #ManufacturingCrisis #FactoryDowntime #Losses"),
            ("Startup shuts down operations after burning through seed capital without achieving product-market fit.", "#StartupPostmortem #BusinessFailure #VC #LessonsLearned")
        ],
        "Neutral": [
            ("Central bank leaves target policy interest rate unchanged following two-day monetary policy committee meeting.", "#CentralBank #MonetaryPolicy #InterestRates #Economy"),
            ("Securities exchange releases quarterly corporate filing calendar for publicly listed multinational enterprises.", "#FinancialFilings #EarningsSeason #WallStreet #Stocks"),
            ("Economic think tank publishes comprehensive white paper analyzing global foreign direct investment flows.", "#EconomicResearch #GlobalTrade #Investment #Policy"),
            ("Merger and acquisition transaction guidelines updated by international trade compliance regulatory body.", "#MergersAndAcquisitions #CorporateLaw #RegulatoryCompliance #Business"),
            ("Monthly commercial freight volume metrics show stable shipping activity along major continental logistics corridors.", "#FreightIndex #Logistics #SupplyChain #TradeData"),
            ("Quarterly labor market report tracks non-farm payroll changes, workforce participation rates, and wage indices.", "#JobsReport #LaborMarket #EconomicData #Employment"),
            ("Annual consumer spending index compiled from retail point-of-sale aggregated transaction volumes.", "#ConsumerIndex #RetailData #MarketResearch #Economics"),
            ("Venture capital fundraising totals for Q3 summarized across software, cleantech, and biotechnology sectors.", "#VentureCapital #StartupFunding #MarketReport #Fintech"),
            ("Treasury bond yield curve comparison across 2-year, 5-year, and 10-year sovereign government maturities.", "#BondYields #FixedIncome #Treasury #FinancialMarkets"),
            ("Corporate governance audit guidelines updated for board independence and audit committee reporting standards.", "#CorporateGovernance #Compliance #BusinessEthics #Audit")
        ]
    },
    "Health": {
        "Positive": [
            ("Clinical trials demonstrate a 75% reduction in disease progression with the new targeted immunotherapy treatment! Breakthrough medicine.", "#MedicalBreakthrough #Oncology #Healthcare #Hope"),
            ("Completed my 100-day morning yoga and mindfulness challenge. Feeling physically stronger, clearer, and more energized than ever!", "#YogaChallenge #Mindfulness #MentalHealth #HealthyLiving"),
            ("Eating balanced whole foods, prioritizing 8 hours of sleep, and daily walking transformed my cardiovascular health biomarkers.", "#Wellness #HealthyHabits #CardioHealth #Nutrition"),
            ("Free preventive health screening camp helped hundreds of underserved community members receive critical diagnoses early.", "#CommunityHealth #PublicHealth #HealthcareForAll #HealthCamp"),
            ("Mental health awareness campaigns are finally reducing the stigma around seeking professional therapy and psychological support.", "#MentalHealthMatters #BreakTheStigma #Wellness #SelfCare"),
            ("Incredible dedication by nursing staff and emergency room physicians during challenging seasonal flu shifts. True healthcare heroes!", "#HealthcareHeroes #Nursing #Gratitude #PublicHealth"),
            ("New wearable biosensors make tracking chronic blood glucose fluctuations effortless and non-invasive for patients.", "#MedTech #DiabetesCare #HealthInnovation #Wearables"),
            ("Celebrated one year of being completely smoke-free! My lung capacity has improved dramatically and I can run 5K easily.", "#SmokeFree #QuitSmoking #HealthMilestone #FitnessJourney"),
            ("Delicious plant-based dinner packed with fiber, colorful veggies, and wholesome nutrients. Healthy food can taste incredible!", "#PlantBased #HealthyRecipes #Nutrition #CleanEating"),
            ("School district introduces daily mindfulness meditation and nutrition education for elementary school students.", "#KidsHealth #NutritionEducation #WellnessInSchools #HealthyFuture")
        ],
        "Negative": [
            ("Astronomical prescription drug prices are forcing elderly patients to ration essential daily life-saving medications.", "#DrugPrices #HealthcareCrisis #PharmaGreed #PatientAdvocacy"),
            ("Burnout among nurses and junior doctors has reached critical levels due to chronic understaffing and 16-hour shifts.", "#HealthcareBurnout #NurseShortage #HospitalCrisis #SaveOurNHS"),
            ("Dealing with debilitating chronic migraine attacks today. Fluorescent lights and screen glare make work nearly impossible.", "#MigraineAwareness #ChronicIllness #PainManagement #HealthStruggles"),
            ("Dangerous dietary supplements with unverified claims continue to circulate unregulated on social media platforms.", "#HealthScam #Misinformation #PublicHealthAlert #ConsumerWarning"),
            ("Insurance denied coverage for medically necessary physical therapy following knee surgery. Fighting the appeal process is exhausting.", "#InsuranceDenial #HealthcareSystem #PatientRights #Frustrated"),
            ("Sedentary lifestyle and excess screen time linked to rising rates of adolescent postural spine deformities and back pain.", "#SedentaryLife #BackPain #ChildHealth #Ergonomics"),
            ("Food deserts in inner-city neighborhoods leave families with no access to affordable fresh produce or nutritious groceries.", "#FoodDesert #HealthInequality #NutritionCrisis #PublicHealth"),
            ("Hospital emergency wait times exceeded 9 hours due to severe triage bottlenecks and lack of available inpatient beds.", "#EmergencyRoom #HospitalWait #HealthcareFail #MedicalCrisis"),
            ("Misleading wellness trends promoting extreme starvation diets are harming vulnerable teenagers' metabolic health.", "#DietCulture #EatingDisorderAwareness #ToxicWellness #MentalHealth"),
            ("Air quality index spiked to hazardous pollution levels, triggering severe asthma flare-ups across the entire metro area.", "#AirPollution #Asthma #RespiratoryHealth #EnvironmentalHealth")
        ],
        "Neutral": [
            ("World Health Organization releases updated international dietary guidelines on daily micronutrient reference intakes.", "#WHO #PublicHealth #NutritionGuidelines #HealthResearch"),
            ("Clinical study published in Lancet analyzes longitudinal correlation between sleep duration and cognitive performance.", "#SleepScience #Neurology #CognitiveHealth #MedicalResearch"),
            ("Public health department publishes regional vaccination schedule and clinic locations for seasonal booster shots.", "#Immunization #PublicHealth #HealthNotice #VaccineSchedule"),
            ("Overview of standard heart rate variability metric calculations during resting baseline physiological assessments.", "#Biometrics #WearableTech #Cardiology #HealthMetrics"),
            ("Medical conference symposium covers advances in non-invasive diagnostic ultrasound imaging modalities.", "#Ultrasound #Radiology #MedicalDevices #HealthTech"),
            ("Nutritional breakdown comparing dietary fiber content across various whole grains and legume varieties.", "#NutritionData #Dietetics #HealthyEating #FoodScience"),
            ("Epidemiological surveillance data tracks seasonal incidence rates of upper respiratory viral tract infections.", "#Epidemiology #InfectiousDisease #Surveillance #PublicHealth"),
            ("Guidelines for ergonomic desk setup, monitor distance, and chair lumbar adjustment to minimize musculoskeletal strain.", "#Ergonomics #WorkplaceHealth #OccupationalSafety #Wellness"),
            ("Pharmaceutical research laboratory begins phase 2 clinical pharmacokinetic evaluation of novel oral antiviral compound.", "#ClinicalTrials #Pharmacology #PharmaResearch #MedicalScience"),
            ("Health ministry publishes statistical bulletin on life expectancy trends and regional demographic aging indicators.", "#Demographics #PublicHealth #HealthStatistics #LifeExpectancy")
        ]
    },
    "Entertainment": {
        "Positive": [
            ("The cinematography in this new sci-fi film is a breathtaking visual masterpiece! Stunning practical effects and profound storytelling.", "#MovieReview #Cinema #Masterpiece #MustWatch"),
            ("Attended an unforgettable live acoustic concert tonight. The vocals were chillingly beautiful and the crowd energy was pure magic!", "#LiveMusic #ConcertVibes #MusicLover #Acoustic"),
            ("Binge-watched the entire new mystery series in one weekend. Every episode kept me guessing until the ingenious final twist!", "#TVSeries #BingeWatch #MysteryThriller #GreatTV"),
            ("The soundtrack composed for this indie video game is an orchestral triumph. Deserves every gaming award this year!", "#GameAudio #IndieGame #Soundtrack #GamingCommunity"),
            ("Thrilled to see indie filmmakers winning top honors at the international festival. Original, bold storytelling always triumphs!", "#IndieFilm #FilmFestival #Filmmaking #Director"),
            ("The animated feature film achieved a breathtaking balance of humor, heart, and visual splendor. Loved every second of it!", "#Animation #MovieNight #FamilyFilm #Storytelling"),
            ("Remarkable theatre performance tonight! The lead actors brought Shakespearean prose to vivid, emotional life on stage.", "#Theatre #Shakespeare #StagePlay #ActingCraft"),
            ("Discovered a phenomenal new indie jazz record. Rich trumpet harmonies, smooth rhythms, and sublime musicianship.", "#JazzMusic #NowListening #VinylCollection #NewMusic"),
            ("The comedy special had the entire auditorium roaring with laughter from beginning to end. Brilliant comedic timing and witty writing!", "#ComedySpecial #StandUp #Laughter #GoodVibes"),
            ("Fantastic museum retrospective showcasing 50 years of progressive cinematic production design and costume artistry.", "#FilmHistory #CostumeDesign #ArtExhibition #CinemaCulture")
        ],
        "Negative": [
            ("Terrible, disjointed movie sequel that relied entirely on nostalgia and cheap fan service while butchering core characters.", "#MovieReview #BadSequel #Disappointed #BoxOfficeFlop"),
            ("Concert venue acoustics were abysmal. Muddy bass overpowered vocals and ticket prices were shockingly exorbitant.", "#ConcertReview #BadAcoustics #Overpriced #Ripoff"),
            ("Ruined an entire television series with an uninspired, rushed final episode full of unresolved plot holes. Total waste of time.", "#SeriesFinale #DisappointingEnding #PlotHoles #TVFail"),
            ("Another uninspired remake that nobody asked for. Hollywood is completely out of original, innovative creative ideas.", "#Hollywood #LazyRemakes #LackOfCreativity #Cinema"),
            ("Video game launched in an unplayable, broken state filled with game-breaking bugs and micro-transaction traps.", "#GamingFail #BrokenGame #BuggyLaunch #DisappointedGamer"),
            ("Streaming service jacks up subscription fee by 30% while simultaneously removing hundreds of classic films and series.", "#StreamingGreed #PriceHike #CancelSubscription #ConsumerLoss"),
            ("The pacing dragged on for three excruciating hours with monotonous dialogue and zero emotional payoff. Do not recommend.", "#FilmReview #BoringMovie #Overrated #SaveYourTime"),
            ("Awful ticket scalping bots bought out every concert ticket in 10 seconds. True fans are left stranded or extorted.", "#TicketScalpers #LiveNation #ConcertScam #FrustratedFans"),
            ("Aggressive intrusive pop-up ads every 3 minutes completely destroy the viewing immersion on free streaming tiers.", "#AdOverload #BadUX #StreamingService #Annoying"),
            ("Casting decisions felt completely mismatched and the dialogue was stiff, unnatural, and devoid of chemistry.", "#BadActing #CastingFail #MovieCritique #PoorDialogue")
        ],
        "Neutral": [
            ("Official screening schedule and ticket reservation window for international film festival announced today.", "#FilmFestival #CinemaSchedule #Screening #FilmNews"),
            ("Box office tracking metrics report weekend theatrical gross totals across top ten domestic and foreign theatrical releases.", "#BoxOffice #MovieGross #FilmIndustry #Theatrical"),
            ("Soundtrack recording credits and orchestrator notes published for upcoming animated feature release.", "#FilmScore #Soundtrack #MusicIndustry #Credits"),
            ("Streaming network announces quarterly content additions and scheduled license expirations for the upcoming month.", "#StreamingGuide #ContentLineup #TVGuide #EntertainmentNews"),
            ("Video game developer studio publishes patch 1.04 balance notes addressing character abilities and collision geometry.", "#PatchNotes #GameDev #GamingUpdate #BalancePatch"),
            ("Television academy reveals eligibility guidelines and nomination ballot dates for annual broadcasting awards.", "#EmmyAwards #Television #Broadcasting #AwardSeason"),
            ("Musicians union negotiates standardized residual framework for digital interactive streaming reproductions.", "#MusicRoyalty #Copyright #EntertainmentLaw #FairCompensation"),
            ("Interview series explores background and historical influences behind traditional stop-motion animation craft.", "#StopMotion #AnimationCraft #BehindTheScenes #Filmmaking"),
            ("Cinematic camera equipment manufacturer releases technical specifications for new 8K digital cinema sensor.", "#CinemaTech #Cinematography #CameraGear #Filmmaker"),
            ("Annual pop culture convention releases guest appearance roster and panel discussion timetable.", "#FanConvention #ComicCon #PopCulture #Panels")
        ]
    },
    "Sports": {
        "Positive": [
            ("What a breathtaking stoppage-time comeback goal to secure the championship trophy! Unstoppable spirit and legendary team effort!", "#Champions #Victory #MatchDay #NeverGiveUp"),
            ("New world record set in the 100m sprint! Pure explosive speed, flawless athletic mechanics, and history made on the track.", "#WorldRecord #Athletics #Sprint #HistoricMoment"),
            ("Phenomenal sportsmanship shown today when the player stopped the attack to assist an injured rival opponent on the pitch.", "#FairPlay #Sportsmanship #Respect #TrueAthlete"),
            ("Our youth academy basketball squad won the regional tournament undefeated! Tremendous coaching and relentless hustle.", "#YouthSports #Basketball #Grassroots #Champs"),
            ("Masterclass defensive performance to shut down the league's top-scoring offense. Tactical discipline at its absolute finest!", "#TacticalMastery #CleanSheet #DefenseWinsChampionships #Football"),
            ("Nothing beats the electrifying atmosphere of a packed derby match under floodlights. Pure passion from start to final whistle!", "#DerbyDay #FootballCulture #StadiumAtmosphere #Fans"),
            ("Incredible personal milestone: scored my 500th career goal tonight in front of home supporters. Grateful for this journey!", "#Milestone #CareerGoals #Legend #Grateful"),
            ("Underdog team defied all betting odds and expert predictions to reach the grand final. The beauty of knockout sports!", "#UnderdogStory #CinderellaRun #Tournament #Glory"),
            ("Flawless gymnastics routine with stuck landings across every apparatus. Deserved that gold medal score without doubt!", "#Gymnastics #GoldMedal #Excellence #OlympicDream"),
            ("Community marathon raised over $250,000 for local children's hospitals. Sport uniting people for a noble purpose!", "#CharityRun #RunForACause #CommunitySpirit #Marathon")
        ],
        "Negative": [
            ("Disgraceful refereeing decision in the 94th minute altered the final outcome. Officiating consistency has hit rock bottom.", "#BadRef #Robbed #RefereeingCrisis #VARFail"),
            ("Devastating anterior cruciate ligament injury confirmed for star quarterback. Season-ending blow to team championship hopes.", "#InjuryReport #Heartbroken #SportsNews #CruelBlow"),
            ("Abysmal team performance today. Zero hustle, disjointed passing, and inexcusable defensive blunders from start to finish.", "#DisappointingLoss #PoorTactics #NoEffort #AngryFans"),
            ("Ticket prices for ordinary working-class supporters are sickening. Club owners treat generational loyal fans like cash cows.", "#AgainstModernFootball #TicketGreed #FanRights #FootballOwnership"),
            ("Doping scandal rocks the cycling tour as multiple athletes test positive for banned performance enhancers. Disgusting.", "#DopingScandal #CleanSport #Cheaters #CyclingNews"),
            ("Riotous fan violence outside the arena marred what should have been a celebration of competitive sport. Shameful behavior.", "#FanTrouble #ZeroTolerance #StadiumSafety #Disgraceful"),
            ("Manager’s rigid stubborn tactics exposed once again against top tier opposition. It's time for a leadership change.", "#ManagerOut #TacticalDisaster #Frustrated #ClubInCrisis"),
            ("Heavy rain and waterlogged drainage system caused unplayable pitch conditions, forcing match abandonment mid-game.", "#MatchPostponed #BadPitch #WeatherDelay #Frustrated"),
            ("Contract holdout drama drags on into training camp. Unprofessional ego battles hurting the locker room morale.", "#ContractDispute #Drama #LockerRoomTension #SportsDebate"),
            ("Relegation confirmed after a miserable season plagued by ownership negligence and mismanagement. Darkest day in club history.", "#Relegation #Heartbreak #ClubCrisis #SupportersGrief")
        ],
        "Neutral": [
            ("Official match statistics, possession charts, passing accuracy, and heatmaps published following league fixture.", "#MatchStats #FootballAnalytics #DataInSports #PostMatch"),
            ("Sports league governing body publishes revised schedule and broadcast network assignments for upcoming season.", "#LeagueSchedule #Broadcasting #SportsNews #Fixtures"),
            ("Player transfer window closes tonight at midnight. Summary of registered incoming and outgoing player transfers.", "#TransferDeadline #TransferNews #SportsBusiness #RosterUpdate"),
            ("Tournament bracket seedings and opening round match pairings announced by committee officials.", "#TournamentBracket #Seedings #Matchups #DrawResults"),
            ("Biomechanics research paper examines kinetic energy transfer in baseball pitching pitching delivery mechanics.", "#SportsScience #Biomechanics #PitchingAnalytics #AthleticPerformance"),
            ("League concussion protocol review protocol procedures presented at annual athletic trainers association meeting.", "#PlayerSafety #ConcussionProtocol #SportsMedicine #Safety"),
            ("Official timing and scoring split interval data verified by international track and field federation delegates.", "#TimingData #TrackAndField #OfficialResults #Athletics"),
            ("Stadium capacity renovations and accessibility upgrades scheduled for completion prior to winter championship.", "#StadiumRenovation #Infrastructure #VenueUpdate #SportsManagement"),
            ("Pre-season fitness camp training regimens and aerobic threshold testing protocols detailed by coaching staff.", "#PreSeason #FitnessTesting #Conditioning #Coaching"),
            ("Referee association convenes annual seminar on VAR implementation standards and handball interpretation rules.", "#VAR #RulesOfTheGame #Officiating #SportsGovernance")
        ]
    },
    "Climate": {
        "Positive": [
            ("Renewable wind and solar energy generated over 65% of regional grid electricity for the entire month! Milestone for transition.", "#CleanEnergy #Renewables #SolarPower #ClimateAction"),
            ("Volunteers planted over 50,000 native mangrove saplings along coastal wetlands to defend against storm surges and restore biodiversity.", "#Reforestation #Mangroves #CoastalDefense #OceanConservation"),
            ("Community banned single-use plastic grocery bags and replaced them with compostable alternatives. Small steps create massive change!", "#PlasticFree #ZeroWaste #EcoFriendly #Sustainability"),
            ("Solar panel installation costs have plummeted by 85% over the past decade, making rooftop green energy accessible to homeowners.", "#SolarEnergy #EnergyTransition #GreenEconomy #SustainableLiving"),
            ("Global ocean treaty ratified by participating nations to establish marine protected zones across high seas waters.", "#OceanTreaty #MarineProtection #Conservation #BluePlanet"),
            ("Electric bus fleet deployed across metropolitan public transit system, cutting municipal diesel emissions significantly.", "#ElectricBuses #CleanTransit #EcoCity #SustainableTransport"),
            ("Innovative carbon-neutral concrete formula enters commercial production, slashing embodied building emissions by 40%.", "#GreenBuilding #CarbonNegative #SustainableTech #Innovation"),
            ("Local river restoration project successfully returned wild salmon spawning runs after three decades of industrial cleanup.", "#RiverRestoration #Biodiversity #EcosystemRecovery #NatureWins"),
            ("Community composting initiative diverted 1,200 tons of organic food waste away from municipal landfills this year.", "#Composting #ZeroWaste #SoilHealth #CircularEconomy"),
            ("Indigenous-led land stewardship initiative protects 2 million acres of critical boreal forest from commercial clear-cutting.", "#IndigenousLeadership #ForestProtection #ConservationHero #Biodiversity")
        ],
        "Negative": [
            ("Catastrophic wildfire triggered by historic heatwave destroys 200,000 hectares of pristine ecological forest habitats.", "#Wildfires #ClimateEmergency #ExtremeWeather #LossAndDamage"),
            ("Microplastic pollution detected in deep oceanic trenches and Antarctic snow samples. Our global plastic crisis is terrifying.", "#PlasticCrisis #OceanPollution #Microplastics #EcologicalEmergency"),
            ("Fossil fuel subsidies hit an unprecedented all-time high despite urgent international climate agreements and pledges.", "#FossilSubsidies #ClimateInaction #EndFossilFuels #PolicyFailure"),
            ("Severe prolonged agricultural drought forces farmers to abandon crops as major regional reservoir water levels drop to 12%.", "#Drought #WaterCrisis #AgricultureFail #ClimateImpacts"),
            ("Illegal deforestation rates in tropical rainforest reserves surged this quarter due to lack of environmental law enforcement.", "#Deforestation #RainforestCrisis #BiodiversityLoss #Ecocide"),
            ("Bleaching event impacts 70% of coral reef tracts as marine heatwave ocean temperatures shatter historic records.", "#CoralBleaching #OceanHeatwave #ReefCrisis #MarineEcology"),
            ("Air quality index hits emergency hazardous levels as smog and particulate matter choke millions of city residents.", "#SmogCrisis #AirPollution #PublicHealthHazard #ToxicAir"),
            ("Corporate greenwashing campaign exposed: energy giant spent 95% of capital expenditure on oil drilling while claiming to be green.", "#Greenwashing #ExposePolluters #ClimateTruth #CorporateAccountability"),
            ("Rising sea level inundation threatens low-lying coastal island communities with imminent permanent displacement.", "#SeaLevelRise #ClimateRefugees #IslandNations #LossAndDamage"),
            ("Methane leaks from neglected extraction pipelines continue unchecked, accelerating near-term atmospheric warming rates.", "#MethaneLeak #ClimateCrisis #PipelineNeglect #GreenhouseGas")
        ],
        "Neutral": [
            ("Intergovernmental meteorological panel releases annual global mean surface temperature anomaly dataset.", "#ClimateData #TemperatureRecord #ScienceReport #Meteorology"),
            ("Satellite telemetry tracks polar sea ice extent variations during seasonal freeze and melt cycles.", "#PolarScience #SeaIce #RemoteSensing #ClimateResearch"),
            ("National atmospheric observatory publishes monthly Mauna Loa atmospheric carbon dioxide concentration measurements.", "#CO2Levels #KeelingCurve #AtmosphericScience #ClimateMetrics"),
            ("Research symposium addresses regional hydrological cycle impacts and river basin water management strategies.", "#Hydrology #WaterResources #ClimateScience #Symposium"),
            ("Technical assessment examines life cycle greenhouse gas emissions across utility-scale battery energy storage systems.", "#LifeCycleAssessment #BatteryStorage #EnergyAnalysis #CleanTech"),
            ("Regional forestry service completes annual aerial survey documenting timber canopy cover and tree mortality rates.", "#Forestry #CanopySurvey #Ecology #EnvironmentalMonitoring"),
            ("Oceanographic institute deploys autonomous deep-sea Argo floats to measure seawater salinity and thermal profiles.", "#Oceanography #ArgoFloats #MarineScience #DataCollection"),
            ("Environmental regulatory agency issues updated particulate emission thresholds for industrial manufacturing facilities.", "#EnvironmentalPolicy #AirQualityStandards #Regulation #Compliance"),
            ("International summit draft text outlines proposed carbon offset accounting frameworks and trading mechanisms.", "#CarbonMarkets #ClimateFinance #PolicyDraft #COP"),
            ("Comprehensive study compares soil carbon sequestration rates across varied agricultural tilling methodologies.", "#SoilScience #Agroecology #CarbonSequestration #AgricultureResearch")
        ]
    },
    "Education": {
        "Positive": [
            ("Thrilled to announce that our university research paper on automated machine learning evaluation was accepted at top conference!", "#AcademicTwitter #Research #MachineLearning #PhDLife"),
            ("Free online coding bootcamps are breaking barriers and enabling students from non-traditional backgrounds to land tech roles.", "#EdTech #CodingBootcamp #LearnToCode #DiversityInTech"),
            ("Our public library opened a free digital media maker space equipped with 3D printers and microcontrollers for young learners.", "#PublicLibraries #Makerspace #STEMEducation #Community"),
            ("Dedicated teachers who stay after class to tutor struggling students deserve the highest praise and compensation in society.", "#TeacherAppreciation #EducationMatters #InspiringTeachers #Dedication"),
            ("First-generation college student graduating summa cum laude today! Grateful to my immigrant parents for their endless sacrifices.", "#FirstGen #GraduationDay #HigherEd #FamilyPride"),
            ("Interactive gamified mathematics curriculum boosted student comprehension and test scores by 25% across pilot schools.", "#EdTechInnovation #MathEducation #Gamification #StudentSuccess"),
            ("Scholarship foundation awarded full university tuition funding to 50 brilliant low-income high school scholars.", "#Scholarships #CollegeAccess #EducationForAll #Empowerment"),
            ("Hands-on robotics club won regional engineering championship. Amazing to witness teenagers designing complex micro-controllers!", "#RoboticsClub #STEM #EngineeringKids #YouthInnovation"),
            ("Open educational resources (OER) saved university students over $500,000 in textbook expenses this semester alone.", "#OER #OpenEducation #AffordableCollege #StudentAdvocacy"),
            ("University opens state-of-the-art interdisciplinary research hub combining artificial intelligence, ethics, and social science.", "#HigherEducation #Interdisciplinary #AIethics #InnovationHub")
        ],
        "Negative": [
            ("Exorbitant college tuition fees combined with predatory student loan interest rates are crushing an entire generation.", "#StudentDebt #HigherEdScam #TuitionCrisis #DebtRelief"),
            ("Public school funding cuts forced our district to eliminate arts, music, and physical education programs. Disgraceful.", "#DefundSchools #SaveTheArts #PublicEducation #Underfunded"),
            ("Overcrowded classrooms with 38 students per teacher make individualized instruction and learning support impossible.", "#OvercrowdedClassrooms #TeacherBurnout #EducationCrisis #StudentNeeds"),
            ("Standardized testing mania stresses young elementary children without measuring actual critical thinking or creativity.", "#StandardizedTesting #TestStress #EducationReform #RethinkTesting"),
            ("Severe nationwide shortage of qualified special education teachers leaves vulnerable neurodivergent students neglected.", "#SpecialEducation #TeacherShortage #SpecialNeeds #InclusionFail"),
            ("College bookstore charging $350 for a single-semester access code that cannot be resold or shared is pure extortion.", "#TextbookRipoff #CollegeCosts #StudentStruggles #AccessCodeGouge"),
            ("Outdated university curriculum teaches obsolete programming languages and zero practical modern cloud architecture skills.", "#OutdatedCurriculum #HigherEdReform #PracticalSkills #TechEducation"),
            ("Unpaid internships that require 40 hours a week exclude talented students who must work paying jobs to survive.", "#UnpaidInternships #FairPay #SocialMobility #LaborExploitation"),
            ("Academic burnout among graduate PhD students is ignored by toxic tenure-track advisory cultures.", "#PhDStress #GradSchoolBurnout #MentalHealthInAcademia #ToxicAdvising"),
            ("Rural school districts continue to lack reliable high-speed broadband internet access, creating a deep digital divide.", "#DigitalDivide #RuralEducation #BroadbandAccess #Inequality")
        ],
        "Neutral": [
            ("University admissions office announces deadlines and standardized testing optional policies for upcoming application cycle.", "#Admissions #CollegeApps #HigherEducation #AdmissionsNews"),
            ("State education board publishes updated curriculum standards framework for secondary school natural sciences.", "#Curriculum #ScienceEducation #K12 #EducationPolicy"),
            ("Comparative survey analyzes student retention rates across online synchronous versus asynchronous learning modalities.", "#DistanceLearning #EdTechResearch #HigherEd #Pedagogy"),
            ("Annual academic library conference convenes to examine open access repository preservation strategies.", "#AcademicLibraries #OpenAccess #ScholarlyCommunication #LibraryScience"),
            ("University registrar releases final examination timetable and room scheduling allocations for fall semester.", "#ExamSchedule #UniversityNews #AcademicCalendar #Registrar"),
            ("Research paper examines statistical correlation between early childhood reading intervention and vocabulary acquisition.", "#ReadingResearch #Literacy #ChildDevelopment #EducationStudy"),
            ("National education statistics center publishes digest of higher education enrollment figures and demographic tables.", "#NCES #EducationData #Demographics #InstitutionalResearch"),
            ("Symposium discusses institutional accreditation standards and continuous quality improvement rubrics.", "#Accreditation #HigherEdQuality #Assessment #InstitutionalEffectiveness"),
            ("Workshop for primary educators provides training in experiential project-based learning lesson plan development.", "#ProjectBasedLearning #TeacherPD #ProfessionalDevelopment #Pedagogy"),
            ("Study tracks post-graduation career placement and employment outcome metrics across undergraduate degree majors.", "#GraduateOutcomes #CareerServices #EmploymentTrends #HigherEdData")
        ]
    },
    "Lifestyle": {
        "Positive": [
            ("Spent the morning brewing pour-over Ethiopian coffee while listening to the birds in my garden. Beautiful peaceful morning.", "#MorningVibes #CoffeeLover #PeacefulMoments #MindfulLiving"),
            ("Decluttered my living space using minimalist principles. Less physical clutter translates into so much mental clarity!", "#Minimalism #Decluttering #SimpleLiving #HomeOrganization"),
            ("Cooked an exquisite homemade artisan sourdough loaf with crispy golden crust and airy crumb. Baking is pure therapy!", "#Sourdough #HomeBaker #BreadMaking #BakingTherapy"),
            ("Breathtaking weekend hiking getaway through pine forest trails. Reconnecting with nature always restores the soul.", "#NatureLovers #HikingAdventures #GetOutside #ForestTherapy"),
            ("Discovered a charming independent bookstore café downtown. Cozy corners, great literature, and delicious cinnamon rolls.", "#BookLovers #BookstoreCafe #CozyVibes #ShopLocal"),
            ("Celebrated 10 years of blissful marriage with my soulmate today. Grateful for every single laugh and shared memory.", "#Anniversary #Soulmate #LoveAndGratitude #LifeJourney"),
            ("Started an herb garden on my apartment balcony. Fresh basil, mint, and rosemary growing vigorously in the morning sun!", "#UrbanGardening #BalconyGarden #GreenThumb #Herbs"),
            ("Sunday meal prep completed for the entire week! Fresh, wholesome macro-balanced meals ready to fuel my daily workouts.", "#MealPrep #HealthyLiving #FitnessLifestyle #TimeManagement"),
            ("Grateful for slow lazy weekend mornings with warm tea, soft blankets, and zero scheduled Zoom commitments.", "#WeekendVibes #SlowLiving #SelfCare #RestAndRecharge"),
            ("Volunteered at the animal rescue shelter this afternoon. Walking playful shelter rescue pups brings boundless happiness!", "#AnimalRescue #AdoptDontShop #DogLover #ShelterPets")
        ],
        "Negative": [
            ("Flight delayed by 7 hours with zero information, lost baggage, and rude customer service desk staff. Worst airline experience ever.", "#AirlineFail #FlightDelay #LostBaggage #TravelNightmare"),
            ("Neighbor decided to start loud power drilling at 6:00 AM on Sunday morning. Absolute lack of basic human courtesy.", "#BadNeighbors #NoisePollution #Rude #SundayRuined"),
            ("Package was stolen straight from my front doorstep within 15 minutes of delivery. Porch pirates are infuriating.", "#PorchPirates #StolenPackage #Frustrated #PackageTheft"),
            ("Plumbing emergency flooded the kitchen floor right before dinner guests arrived. Stress levels through the roof today.", "#PlumbingDisaster #HomeEmergency #BadLuck #StressfulDay"),
            ("Ordered bespoke dining table 3 months ago; arrived with scratched wood finish and cracked legs. Return process is an ordeal.", "#DamagedGoods #BadDelivery #PoorQuality #CustomerServiceFail"),
            ("Dry cleaner completely ruined my vintage tailored wool overcoat and refused to accept responsibility or offer compensation.", "#CustomerDisaster #RuinedClothes #DryCleanerFail #Unprofessional"),
            ("Endured terrible food poisoning after eating takeout sushi last night. Agonizing stomach cramps all day long.", "#FoodPoisoning #SickDay #NeverAgain #Unsanitary"),
            ("Hotel reservation cancelled without warning due to overbooking, leaving us stranded in an unfamiliar city late at night.", "#HotelScam #Overbooked #TravelNightmare #BadHospitality"),
            ("Traffic gridlock caused by poorly timed highway construction added 2.5 miserable hours to my evening commute.", "#TrafficJam #CommuteHell #Gridlock #Frustrated"),
            ("Appliance repair technician arrived 4 hours late, charged a ridiculous diagnostic fee, and didn't even fix the problem.", "#RepairFail #Ripoff #CustomerComplaint #UnreliableService")
        ],
        "Neutral": [
            ("City public transit authority publishes holiday timetable adjustments for commuter rail and regional buses.", "#TransitSchedule #CommuterAlert #CityLife #PublicTransport"),
            ("Municipal farmers market operating hours and vendor stall directory updated for the autumn harvest season.", "#FarmersMarket #LocalProduce #HarvestSeason #CommunityNotice"),
            ("Review of indoor humidity levels and air purifier filter replacement intervals for residential living spaces.", "#HomeCare #AirPurifier #IndoorAirQuality #HomeTips"),
            ("Overview of standard fabric care symbols, water wash temperature codes, and tumble dry cycle designations.", "#LaundryGuide #FabricCare #HomeHacks #ClothingCare"),
            ("Urban planning commission hosts public informational hearing regarding bike lane network connectivity expansions.", "#UrbanPlanning #BikeLanes #CityDevelopment #CivicEngagement"),
            ("Summary of seasonal pantry staple storage temperatures and recommended shelf-life inspection timelines.", "#PantryOrganization #FoodStorage #KitchenManagement #HomeEconomics"),
            ("Comparison of natural lighting orientation versus artificial task illumination across remote home office workspaces.", "#HomeOffice #WorkspaceDesign #Lighting #InteriorDesign"),
            ("Municipal botanical conservatory announces open admission hours and scheduled guided greenhouse tours.", "#BotanicalGarden #Horticulture #Greenhouse #CityGuide"),
            ("Guide to seasonal indoor houseplant watering frequency adjustments during cooler ambient temperature months.", "#Houseplants #PlantCare #UrbanJungle #GreenLiving"),
            ("Postal service releases domestic and international holiday parcel shipping deadlines and postage rates.", "#PostalService #ShippingDeadlines #MailGuide #HolidayNotice")
        ]
    }
}

def generate_engagement(platform: str, sentiment: str) -> tuple[int, int, int]:
    """Generate realistic, non-negative likes, comments, and shares."""
    # Engagement distributions vary by platform and sentiment (positive/negative often get more engagement than neutral)
    multiplier = 1.2 if sentiment in ["Positive", "Negative"] else 0.8
    
    if platform == "Instagram":
        likes = int(random.lognormvariate(5.5, 1.1) * multiplier)
        comments = int(likes * random.uniform(0.02, 0.08))
        shares = int(likes * random.uniform(0.01, 0.05))
    elif platform == "X":
        likes = int(random.lognormvariate(5.0, 1.2) * multiplier)
        comments = int(likes * random.uniform(0.04, 0.15))
        shares = int(likes * random.uniform(0.05, 0.25))  # Retweets are high
    elif platform == "YouTube":
        likes = int(random.lognormvariate(6.0, 1.0) * multiplier)
        comments = int(likes * random.uniform(0.05, 0.12))
        shares = int(likes * random.uniform(0.01, 0.04))
    else:  # Facebook
        likes = int(random.lognormvariate(4.8, 1.1) * multiplier)
        comments = int(likes * random.uniform(0.03, 0.10))
        shares = int(likes * random.uniform(0.03, 0.12))
    
    # Cap maximums to realistic values and guarantee non-negative
    likes = max(0, min(likes, 25000))
    comments = max(0, min(comments, 5000))
    shares = max(0, min(shares, 3500))
    return likes, comments, shares

def generate_dataset(output_path: Path, count: int = 1200):
    """Generate sample dataset of synthetic social media posts."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    start_date = datetime(2024, 7, 1, 8, 0, 0)
    end_date = datetime(2024, 9, 30, 23, 59, 59)
    total_seconds = int((end_date - start_date).total_seconds())
    
    rows = []
    
    # Target sentiment distribution: 40% Positive, 30% Negative, 30% Neutral
    sentiments = ["Positive"] * 4 + ["Negative"] * 3 + ["Neutral"] * 3
    
    for i in range(1, count + 1):
        post_id = f"POST_{i:04d}"
        platform = random.choices(PLATFORMS, weights=PLATFORM_WEIGHTS)[0]
        username = random.choice(USERNAMES)
        topic = random.choice(TOPICS)
        sentiment = random.choice(sentiments)
        
        # Pick template from the topic and sentiment
        template_list = TEMPLATES[topic][sentiment]
        base_text, base_hashtags = random.choice(template_list)
        
        # Slight variation or personalization
        post_text = base_text
        hashtags = base_hashtags
        
        likes, comments, shares = generate_engagement(platform, sentiment)
        
        # Random timestamp within window
        random_seconds = random.randint(0, total_seconds)
        post_date = (start_date + timedelta(seconds=random_seconds)).strftime("%Y-%m-%d %H:%M:%S")
        
        rows.append({
            "post_id": post_id,
            "platform": platform,
            "username": username,
            "post_text": post_text,
            "hashtags": hashtags,
            "likes": likes,
            "comments": comments,
            "shares": shares,
            "post_date": post_date,
            "topic": topic
        })
    
    # Write to CSV
    fieldnames = [
        "post_id", "platform", "username", "post_text", "hashtags",
        "likes", "comments", "shares", "post_date", "topic"
    ]
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"Successfully generated {len(rows)} records at: {output_path}")

if __name__ == "__main__":
    out_file = Path(__file__).resolve().parent.parent / "data" / "social_media_posts.csv"
    generate_dataset(out_file, count=1200)
