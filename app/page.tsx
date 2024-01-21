"use client";

import { useState } from "react";
const AccordionItem = ({
  index,
  title,
  content,
}: {
  index: number;
  title: string;
  content: string;
}) => {
  const [isOpen, setIsOpen] = useState(false);

  const handleToggle = () => {
    setIsOpen((prev) => !prev);
  };

  return (
    <div className="collapse collapse-arrow bg-base-200 w-5/6 mx-auto ">
      <input
        type="radio"
        name={`my-accordion-${index}`}
        checked={isOpen}
        onChange={handleToggle}
      />
      <div
        className="collapse-title text-xl font-medium"
        onClick={handleToggle}
      >
        {title}
      </div>
      <div
        className="collapse-content"
        style={{ maxHeight: isOpen ? "1000px" : "0px" }}
      >
        <p>{content}</p>
      </div>
    </div>
  );
};

export default function Home() {
  const faqs = [
    {
      title: "Q: What is prakriti in Ayurveda?",
      content:
        "A: Prakriti refers to an individual's inherent constitution or Ayurvedic blueprint, comprising the unique balance of Vata, Pitta, and Kapha doshas that influence physical and mental characteristics.",
    },
    {
      title: "Q: How can I determine my prakriti?",
      content:
        "A: Discovering your prakriti is easy with our interactive chatbot. Answer a few simple questions about your lifestyle, preferences, and body characteristics, and receive personalized insights into your Ayurvedic constitution.",
    },
    {
      title: "Q: Why is prakriti important for my well-being?",
      content:
        "A: Understanding your prakriti enables personalized wellness recommendations. It helps in creating a balanced lifestyle, preventing imbalances, and promoting overall health and vitality tailored to your unique needs.",
    },
    {
      title: "Q: Can prakriti change over time?",
      content:
        "A: While your prakriti remains fairly constant, external factors, lifestyle, and age may influence temporary imbalances. Regular Ayurvedic practices can help bring the doshas back into harmony.",
    },
    {
      title: "Q: Are Ayurvedic recommendations based on prakriti effective?",
      content:
        "A: Yes, Ayurvedic recommendations tailored to your prakriti are effective. They provide a holistic approach to well-being, considering physical, mental, and emotional aspects, promoting sustainable health benefits.",
    },
  ];
  return (
    <div className="container mx-auto w-full">
      <div className="grid sm:grid-cols-1 lg:grid-cols-3">
        <div className="pt-20 px-7 space-y-4 lg:col-span-2 sm:col-span-1">
          <div className="order-1 lg:w-2/3">
            <div className="text-left font-bold text-6xl mb-5">
              Discover your Prakriti
            </div>
            <div className="text-left font-serif text-3xl mb-5">
              Balance and Harmony with{" "}
              <span className="text-green-600">Ayurveda</span>
            </div>
            <a href="https://console.dialogflow.com/api-client/demo/embedded/87240b21-7222-4d83-b612-6d47a56f73b1">
              <button className="btn btn-outline btn-accent">
                Use Ayur Bot🌿
              </button>
            </a>
            <div className="divider"></div>
            <div className="font-sans text-2xl max-w-3xl text-justify">
              Welcome to our Ayurvedic Prakriti Prediction platform. where
              ancient wisdom meets modern technology to guide you on your
              journey to wellness. Ayurveda, the ancient holistic healing
              system, recognizes that each person is unique. with a distinct
              constitution known as Prakriti. Understanding your Prakriti can
              help you make lifestyle choices that promote balance, health, and
              well-being.
            </div>
          </div>
        </div>
        <div className="order-2 py-20 lg:ml-[-5rem]">
          <iframe
            height="500"
            width="500"
            src="https://console.dialogflow.com/api-client/demo/embedded/87240b21-7222-4d83-b612-6d47a56f73b1"
            style={{
              boxShadow:
                "0 10px 20px rgba(0, 0, 0, 0.2), 0 6px 6px rgba(0, 0, 0, 0.15)",
              borderRadius: "8px",
            }}
          ></iframe>
        </div>
      </div>
      <hr className="my-12 h-px border-t-0 bg-transparent bg-gradient-to-r from-transparent via-neutral-500 to-transparent opacity-25 dark:opacity-100" />
      <div className="pt-7 px-7 space-y-4">
        <div className="font-bold text-4xl underline mx-auto w-[max-content]">
          Frequently Asked Questions (FAQs)
        </div>
        {faqs.map((faq, index) => (
          <AccordionItem
            key={index}
            index={index}
            title={faq.title}
            content={faq.content}
          />
        ))}
      </div>
    </div>
  );
}
