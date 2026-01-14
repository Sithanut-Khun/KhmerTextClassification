import React, { useState } from 'react';
import { Client } from "@gradio/client";
import khunSithanutPhoto from './assets/khun_sithanut.jpg';
import Ringo from './assets/e20211040.jpg';
import siveeu from './assets/eng_sive_eu.jpg';
import bunRatnatepy from './assets/bun_ratnatepy.jpg';

// --- CONFIGURATION ---
const HF_SPACE_URL = "https://thanut003-khmer-text-classifier-api.hf.space";

// --- TEAM MEMBERS DATA ---
const MEMBERS = [
  {
    name: "Khun Sithanut",
    role: "Data Scientist",
    email: "sithanutkhun@gmail.com",
    linkedin: "https://www.linkedin.com/in/khun-sithanut-a71945239?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app",
    photo: khunSithanutPhoto
  },
  {
    name: "KHEM Lyhourt",
    role: "Data Scientist",
    email: "khemlyhourtt@gmail",
    linkedin: "https://www.linkedin.com/in/khem-lyhourt-502032352/",
    photo: Ringo
  },
  {
    name: "Bun Ratnatepy",
    role: "Data Scientist",
    email: "bunratnatepy@gmail.com",
    linkedin: "https://www.linkedin.com/in/bun-ratnatepy-5859342b6/",
    photo: bunRatnatepy
  },
  {
    name: "Eng Sive Eu",
    role: "Data Scientist",
    email: "engseaveeu@gmail.com",
    linkedin: "https://www.linkedin.com/in/eng-seav-eu-b53184332/",
    photo: siveeu
  },
  {
    name: "Chhran Moses",
    role: "Data Scientist",
    email: "chhranmoses543@gmail.com",
    linkedin: "#",
    photo: "https://api.dicebear.com/7.x/avataaars/svg?seed=Max"
  },
  {
    name: "Lun Chan Poly",
    role: "Data Scientist",
    email: "student6@e.t-ams.edu.kh",
    linkedin: "#",
    photo: "https://api.dicebear.com/7.x/avataaars/svg?seed=Zoe"
  },
];

// --- TRANSLATIONS ---
const UI_TEXT = {
  en: {
    title: "Clarify AI",
    subtitle: "Khmer Edition",
    btnAbout: "About Us",    
    btnBack: "Back",
    activeModel: "Active Model",
    inputLabel: "Input Khmer Text",
    placeholder: "Paste your Khmer article content here...",
    classifyBtn: "Classify Text",
    analyzing: "Analyzing...",
    predictedCategory: "Predicted Category",
    confidence: "Confidence Score",
    probabilities: "Confidence Breakdown",
    history: "Recent History",
    runPrompt: "Run an analysis to see detailed scores.",
    modelInfo: "Prediction based on tokenized keywords.",
    keywords: "Top Key Factors",
    error: "Error connecting to API.",
    aboutTitle: "The Team",
    aboutDesc: "We are Year 5 students from the Department of Applied Mathematics and Statistics, Majoring in Data Science.",
    uni: "Institute of Technology of Cambodia"
  },
  km: {
    title: "Clarify AI",
    subtitle: "សម្រាប់ភាសាខ្មែរ",
    btnAbout: "អំពីពួកយើង",   
    btnBack: "ត្រឡប់",
    activeModel: "ម៉ូដែលដែលកំពុងប្រើ",
    inputLabel: "បញ្ចូលអត្ថបទភាសាខ្មែរ",
    placeholder: "សូមសរសេអត្ថបទរបស់អ្នកនៅទីនេះ...",
    classifyBtn: "វិភាគអត្ថបទ",
    analyzing: "កំពុងវិភាគ...",
    predictedCategory: "ប្រភេទអត្ថបទ",
    confidence: "កម្រិតភាពជាក់លាក់",
    probabilities: "លម្អិតនៃលទ្ធផលប្រូបាប",
    history: "ប្រវត្តិថ្មីៗ",
    runPrompt: "សូមធ្វើការវិភាគដើម្បីមើលលទ្ធផលលម្អិត។",
    modelInfo: "ការទស្សន៍ទាយផ្អែកលើពាក្យគន្លឹះក្នុងអត្ថបទ។",
    keywords: "ពាក្យគន្លឹះសំខាន់ៗ",
    error: "មានបញ្ហាក្នុងការភ្ជាប់ទៅកាន់ API។",
    aboutTitle: "ក្រុមការងារ",
    aboutDesc: "ពួកយើងជានិស្សិតឆ្នាំទី ៥ នៃដេប៉ាតឺម៉ង់គណិតវិទ្យាអនុវត្ត និងស្ថិតិ ជំនាញវិទ្យាសាស្ត្រទិន្នន័យ។",
    uni: "វិទ្យាស្ថានបច្ចេកវិទ្យាកម្ពុជា"
  }
};

const CATEGORY_MAP = {
  'Culture': { en: 'Culture', km: 'វប្បធម៌' },
  'Economic': { en: 'Economic', km: 'សេដ្ឋកិច្ច' },
  'Education': { en: 'Education', km: 'ការអប់រំ' },
  'Environment': { en: 'Environment', km: 'បរិស្ថាន' },
  'Health': { en: 'Health', km: 'សុខភាព' },
  'Politics': { en: 'Politics', km: 'នយោបាយ' },
  'Human Rights': { en: 'Human Rights', km: 'សិទ្ធិមនុស្ស' },
  'Science': { en: 'Science', km: 'វិទ្យាសាស្ត្រ' }
};

function App() {
  const [inputText, setInputText] = useState('');
  const [selectedModel, setSelectedModel] = useState('XGBoost (BoW)');
  const [result, setResult] = useState({ label: null, confidences: null, keywords: [] });
  const [isLoading, setIsLoading] = useState(false);
  const [history, setHistory] = useState([]);
  const [lang, setLang] = useState('en'); 
  const [view, setView] = useState('home'); 

  const t = UI_TEXT[lang]; 

  const handlePredict = async () => {
    if (!inputText.trim()) return;
    setIsLoading(true);
    setResult({ label: null, confidences: null, keywords: [] });

    try {
      const client = await Client.connect(HF_SPACE_URL);
      const response = await client.predict("/predict", [ inputText, selectedModel ]);

      const topLabel = response.data[0].label;
      const confData = response.data[1];
      const keywordsData = response.data[2];

      let cleanConfidences = {};
      if (confData && confData.confidences) {
        confData.confidences.forEach(item => cleanConfidences[item.label] = item.confidence);
      } else if (typeof confData === 'object') {
        cleanConfidences = confData;
      }

      setResult({ label: topLabel, confidences: cleanConfidences, keywords: keywordsData });
      setHistory(prev => [{ text: inputText, label: topLabel, date: new Date() }, ...prev]);

    } catch (err) {
      console.error("API Error:", err);
      alert(t.error);
    } finally {
      setIsLoading(false);
    }
  };

  const getTopConfidence = () => {
    if (!result.confidences) return 0;
    const values = Object.values(result.confidences);
    return (Math.max(...values) * 100).toFixed(1);
  };

  const translateLabel = (label) => {
    if (!label) return "";
    return CATEGORY_MAP[label] ? CATEGORY_MAP[label][lang] : label;
  };

  // Helper to toggle view
  const toggleView = () => {
    setView(prev => prev === 'home' ? 'about' : 'home');
  };

  return (
    <div className={`flex flex-col h-screen w-screen bg-gray-900 text-gray-100 font-sans overflow-hidden ${lang === 'km' ? 'font-khmer' : ''}`}>
      
      {/* HEADER */}
      <header className="h-16 bg-gray-800 border-b border-gray-700 flex items-center justify-between px-6 shrink-0 z-10">
        <div className="flex items-center space-x-6">
          <div className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-indigo-500 cursor-pointer" onClick={() => setView('home')}>
            {t.title}
          </div>
        </div>

        <div className="flex items-center space-x-4">
            {/* 1. THE SMART TOGGLE BUTTON */}
            <button 
                onClick={toggleView}
                className="px-4 py-1.5 text-sm bg-gray-700 hover:bg-gray-600 text-gray-200 rounded-lg transition border border-gray-600 font-medium"
            >
                {view === 'home' ? t.btnAbout : t.btnBack}
            </button>

            {/* 2. LANGUAGE SWITCHER */}
            <div className="flex bg-gray-700 rounded-lg p-1">
                <button onClick={() => setLang('en')} className={`px-3 py-1 text-xs rounded-md transition ${lang === 'en' ? 'bg-indigo-600 text-white shadow' : 'text-gray-400 hover:text-white'}`}>English</button>
                <button onClick={() => setLang('km')} className={`px-3 py-1 text-xs rounded-md transition ${lang === 'km' ? 'bg-indigo-600 text-white shadow' : 'text-gray-400 hover:text-white'}`}>ខ្មែរ</button>
            </div>
        </div>
      </header>

      {/* DYNAMIC VIEW CONTENT */}
      {view === 'home' ? (
        // --- CLASSIFIER VIEW ---
        <div className="flex flex-1 overflow-hidden relative">
            <aside className="w-72 bg-gray-800 border-r border-gray-700 flex flex-col hidden md:flex z-0">
            <div className="p-5 border-b border-gray-700">
                <h3 className="text-xs uppercase tracking-wider text-gray-500 mb-3 font-semibold">{t.activeModel}</h3>
                <select 
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                    className="w-full bg-gray-900 border border-gray-600 rounded-lg text-sm p-3 text-white focus:ring-2 ring-indigo-500 outline-none"
                >
                    {/* <option value="XGBoost">XGBoost</option>
                    <option value="LightGBM">LightGBM</option>
                    <option value="Random Forest">Random Forest</option>
                    <option value="Logistic Regression">Logistic Regression</option>
                    <option value="Linear SVM">Linear SVM</option> */}

                    <option value="XGBoost (BoW)">XGBoost (BoW)</option>
                    <option value="LightGBM (BoW)">LightGBM (BoW)</option>
                    <option value="Random Forest (BoW)">Random Forest (BoW)</option>
                    <option value="Logistic Regression (TF-IDF + SVD)">Logistic Regression (TF-IDF + SVD)</option>
                    <option value="Linear SVM (TF-IDF + SVD)">Linear SVM (TF-IDF + SVD)</option>
                </select>
            </div>
            <div className="flex-1 overflow-y-auto p-4 custom-scrollbar">
                <h3 className="text-xs uppercase tracking-wider text-gray-500 mb-3 font-semibold">{t.history}</h3>
                {history.map((item, idx) => (
                    <div key={idx} onClick={() => setInputText(item.text)} className="p-3 bg-gray-700/40 rounded-lg mb-2 cursor-pointer hover:bg-gray-700 transition">
                        <p className="text-xs text-gray-300 truncate font-medium">"{item.text}"</p>
                        <span className="text-[10px] text-indigo-300 bg-indigo-900/30 px-1.5 py-0.5 rounded mt-1 inline-block">
                        {translateLabel(item.label)}
                        </span>
                    </div>
                ))}
            </div>
            </aside>

            <main className="flex-1 overflow-y-auto p-6 md:p-10 bg-gray-900">
            <div className="max-w-3xl mx-auto space-y-8">
                <div className="bg-gray-800 rounded-2xl shadow-xl border border-gray-700 overflow-hidden">
                    <div className="p-6">
                        <label className="block text-sm font-medium text-gray-400 mb-3 uppercase tracking-wide">{t.inputLabel}</label>
                        <textarea 
                            className="w-full h-40 bg-gray-900 border border-gray-600 rounded-xl p-4 text-gray-200 focus:ring-2 ring-indigo-500 focus:border-transparent outline-none resize-none text-lg" 
                            placeholder={t.placeholder}
                            value={inputText}
                            onChange={(e) => setInputText(e.target.value)}
                        ></textarea>
                        <div className="flex justify-end mt-4">
                            <button 
                                onClick={handlePredict}
                                disabled={isLoading || !inputText.trim()}
                                className="bg-indigo-600 hover:bg-indigo-500 text-white px-8 py-2.5 rounded-lg font-semibold transition-all shadow-lg flex items-center"
                            >
                                {isLoading ? t.analyzing : t.classifyBtn}
                            </button>
                        </div>
                    </div>
                </div>

                {result.label && (
                    <div className="animate-fade-in-up">
                    <div className="bg-gray-800 rounded-2xl shadow-2xl border border-gray-700 overflow-hidden relative">
                        <div className="absolute top-0 left-0 w-2 h-full bg-green-500"></div>
                        <div className="p-8 pl-10">
                            <div className="flex justify-between items-start mb-6">
                                <div>
                                    <h2 className="text-xs text-gray-400 uppercase tracking-widest font-semibold mb-1">{t.predictedCategory}</h2>
                                    <div className="text-4xl md:text-5xl font-extrabold text-white leading-tight">
                                        {translateLabel(result.label)}
                                    </div>
                                </div>
                                <div className="text-right pl-4">
                                    <div className="text-4xl font-bold text-green-400">{getTopConfidence()}%</div>
                                    <div className="text-xs text-gray-500 font-medium uppercase mt-1">{t.confidence}</div>
                                </div>
                            </div>
                            
                            <div className="mb-6">
                                <h3 className="text-xs text-gray-500 uppercase font-semibold mb-2">{t.keywords}</h3>
                                <div className="flex flex-wrap gap-2">
                                {result.keywords && result.keywords.length > 0 ? (
                                    result.keywords.map((kw, i) => (
                                    <span key={i} className="px-3 py-1 bg-gray-900 border border-gray-600 rounded-full text-sm text-indigo-300 font-khmer">
                                        {kw}
                                    </span>
                                    ))
                                ) : (
                                    <span className="text-sm text-gray-600">No specific keywords found.</span>
                                )}
                                </div>
                            </div>

                            <div className="w-full bg-gray-700 rounded-full h-3 mb-2">
                                <div className="bg-green-500 h-full rounded-full transition-all duration-1000" style={{ width: `${getTopConfidence()}%` }}></div>
                            </div>
                        </div>
                    </div>
                    </div>
                )}
            </div>
            </main>

            <aside className="w-80 bg-gray-800 border-l border-gray-700 p-6 hidden lg:block overflow-y-auto">
                {result.confidences ? (
                <div className="animate-fade-in">
                    <h3 className="text-xs uppercase tracking-wider text-gray-500 mb-6 font-semibold">{t.probabilities}</h3>
                    <div className="space-y-4">
                        {Object.entries(result.confidences)
                            .sort(([,a], [,b]) => b - a)
                            .map(([label, score], idx) => (
                                <div key={label}>
                                    <div className="flex justify-between text-sm mb-1">
                                        <span className={idx === 0 ? "text-white font-medium" : "text-gray-400"}>
                                        {translateLabel(label)}
                                        </span>
                                        <span className="text-gray-500">{(score * 100).toFixed(1)}%</span>
                                    </div>
                                    <div className="w-full bg-gray-700 rounded-full h-1.5">
                                        <div className={`h-full rounded-full ${idx === 0 ? 'bg-green-500' : 'bg-gray-600'}`} style={{ width: `${score * 100}%` }}></div>
                                    </div>
                                </div>
                            ))
                        }
                    </div>
                </div>
                ) : (
                <div className="flex flex-col items-center justify-center h-full text-center opacity-40">
                    <p className="text-sm text-gray-400">{t.runPrompt}</p>
                </div>
                )}
            </aside>
        </div>
      ) : (
        // --- ABOUT US VIEW ---
        <div className="flex-1 overflow-y-auto bg-gray-900 p-8 animate-fade-in">
            <div className="max-w-6xl mx-auto text-center mb-12">
                <h1 className="text-4xl font-bold text-white mb-4 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-indigo-500">{t.aboutTitle}</h1>
                <p className="text-lg text-gray-400 max-w-3xl mx-auto leading-relaxed">{t.aboutDesc}</p>
                <p className="text-sm text-indigo-400 mt-4 font-semibold uppercase tracking-wide border-t border-gray-800 pt-4 inline-block">{t.uni}</p>
            </div>

            <div className="max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 pb-12">
                {MEMBERS.map((member, idx) => (
                    <div key={idx} className="group relative bg-gray-800 rounded-2xl p-6 border border-gray-700 transition-all duration-300 hover:scale-105 hover:shadow-2xl hover:shadow-indigo-500/20 hover:border-indigo-500/50">
                        {/* Interactive Background Glow */}
                        <div className="absolute inset-0 bg-gradient-to-br from-indigo-600/0 to-indigo-600/0 group-hover:from-indigo-600/10 group-hover:to-purple-600/10 rounded-2xl transition-all duration-300"></div>
                        
                        <div className="relative z-10 flex flex-col items-center">
                            <div className="w-40 h-40 rounded-full border-4 border-gray-700 group-hover:border-indigo-500 overflow-hidden mb-4 transition-all duration-300 shadow-lg">
                                <img src={member.photo} alt={member.name} className="w-full h-full object-cover" />
                            </div>
                            
                            <h3 className="text-xl font-bold text-white mb-1 group-hover:text-indigo-400 transition-colors">{member.name}</h3>
                            <p className="text-sm text-gray-500 mb-4">{member.role}</p>
                            
                            <div className="w-full space-y-3">
                                <a href={`mailto:${member.email}`} className="flex items-center justify-center space-x-2 w-full py-2 bg-gray-700/50 hover:bg-gray-700 rounded-lg text-sm text-gray-300 transition-colors">
                                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"></path></svg>
                                    <span>{member.email}</span>
                                </a>
                                <a href={member.linkedin} target="_blank" rel="noreferrer" className="flex items-center justify-center space-x-2 w-full py-2 bg-blue-600/20 hover:bg-blue-600/40 text-blue-400 rounded-lg text-sm transition-colors border border-blue-600/30">
                                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/></svg>
                                    <span>LinkedIn Profile</span>
                                </a>
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        </div>
      )}
    </div>
  );
}

export default App;