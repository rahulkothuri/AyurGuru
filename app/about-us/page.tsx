import Logo from "@/public/logo.png";
import Image from 'next/image'
import sdg from '@/public/sdg.png'
export default function AboutUS() {
  return (
    <div className="container mx-auto w-full">
      <div className="grid sm:grid-cols-1 lg:grid-cols-3">
        <div className="pt-20 px-7 space-y-4 col-span-2">
          <div className="order-1 lg:w-2/3">
            <div className="text-left font-bold text-5xl mb-5 underline">
              About us
            </div>
            <div className="font-sans text-xl max-w-3xl text-justify">
              Have you ever yearned to understand the blueprint of your
              well-being? To unlock the secrets of your mind, body, and soul,
              and align your choices with your unique potential? At AyurGuru, we
              believe the key lies in Prakriti, your Ayurvedic constitution, a
              map etched within you since birth. We are not just another
              website. We are visionaries: bridging the gap between ancient
              wisdom and modern technology through the power of{" "}
              <span className="font-bold">
                {" "}
                Natural Language Processing (NLP)
              </span>
              . Our revolutionary platform is home to an intelligent chatbot,
              your personal guide on a journey of self-discovery.
            </div>
            <div className="text-left font-bold text-3xl mb-5 underline pt-3">
              Our Mission:
            </div>
            <div className="font-sans text-xl max-w-3xl text-justify">
              <span className="font-bold">Democratize Ayurveda:</span> We
              believe Prakriti knowledge should be accessible to everyone, not
              just the privileged few. Our NLP chatbot makes Prakriti assessment
              fun, interactive, and approachable
            </div>
            <div className="font-sans text-xl max-w-3xl text-justify">
              <span className="font-bold">Empower with Insights:</span>{" "}
              Understanding your Prakriti is the first step to holistic
              wellness. Our chatbot delves deep into your responses, unveiling
              your dominant Doshas (Vata, Pitta, Kapha) and revealing the
              secrets they hold.
            </div>
            <div className="font-sans text-xl max-w-3xl text-justify">
              <span className="font-bold">Spark a Wellness Revolution:</span> We
              dream of a world where individuals thrive by aligning their
              choices with their Prakriti needs. Our platform is a catalyst for
              conscious living, empowering you to make informed decisions about
              your health, relationships, and career.
            </div>
            <div className="text-left font-bold text-3xl mb-5 underline pt-3">
              The Technology Behind the Magic
            </div>
            <div className="font-sans text-xl max-w-3xl text-justify">
              Our NLP chatbot is no ordinary conversationalist. It{"'"}s a{" "}
              <span className="font-bold">sophisticated AI</span> trained on
              vast amounts of Ayurvedic data and linguistic nuances. As you
              interact, the chatbot analyzes your responses, identifying subtle
              patterns that paint a picture of your unique Prakriti. It{"'"}s like
              having a wise Ayurvedic master whispering the secrets of your
              inner self.
            </div>
            <div className="text-left font-bold text-3xl mb-5 underline pt-3">
              Meet the Team
            </div>
            <div className="font-sans text-xl max-w-3xl text-justify">
              We are a passionate team of students united by a common goal: to
              make Prakriti knowledge accessible and transformative. We believe
              technology can be a powerful tool for good, and we{"'"}re dedicated to
              using it to empower individuals on their wellness journeys.
            </div>
            <div className="text-left font-bold text-3xl mb-5 underline pt-3">
              The Impact We See
            </div>
            <div className="font-sans text-xl max-w-3xl text-justify">
              The stories we hear from our users are our greatest reward. From
              rediscovering forgotten strengths to making life-changing health
              choices, the insights gleaned from Prakriti assessment have had a
              profound impact on countless lives. We are creating a ripple
              effect of wellness, one Prakriti discovery at a time.
            </div>
            <div className="text-left font-bold text-3xl mb-5 underline pt-3">
              Join our community!
            </div>
            <div className="font-sans text-xl max-w-3xl text-justify">
              Explore our website for additional resources, connect with fellow
              Prakriti seekers on our forum, and stay updated on our latest
              developments. Together, let{"'"}s build a world where understanding
              your Prakriti is the foundation for a life well-lived.
            </div>
          </div>

        </div>
        <div className="order-2py-20 lg:ml-[-7rem]">
            <Image
             src = {Logo}
             alt = "logo"
             />
             <Image
             src = {sdg}
             alt = "sdg"
             width = "500"
             /> 
          </div>
      </div>
    </div>
  );
}
