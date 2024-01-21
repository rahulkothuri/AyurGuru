import Image from "next/image";
import banner from "@/public/info/prakriti.png";
export default function Prakriti() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen">
      <div className="max-w-4xl p-4">
        <Image src={banner} alt="Banner" width={1200} height={600} />
        <div className="mt-4">
          <h1 className="text-5xl font-bold">Prakriti</h1>
          <p className="mt-2 text-gray-600">Published on January 21, 2024</p>
          <div className="mt-4 space-y-7">
            {/* Your blog post content goes here */}
            <p className="text-justify ">
              Prakriti is your body constitution / Body type. It is some total
              of one's tridosha percentage (vata, pitta, kapha) in a body. It is
              the key determinant of how one individual is different from other.
              Prakriti has been mentioned in details in Charak Samhita "Vimaan
              sthan chapter 8. It is one of the most important investigation of
              ten investigations regarding patients (Dash vidh pareeksha).
              Prakriti is very important diagnostic tool in Ayurveda. The
              fundamental theory of Ayurveda is the "Tridosh" principle or the
              tri-energies theory. This can also be understood as combination of
              'panchbhutas' (Five Elements) into three doshas. These are
              biological or physiological and physical forces that cater to the
              metabolic functions and structural composition of our body. Kapha
              is responsible for anabolism. Pitta is responsible for metabolism.
              Vata is responsible for catabolism. The balance of these
              tri-energies is known as a state of health and their imbalance is
              disease. Anything which restores this balance is good for health.
              So specific diet, exercise, behavior or medicine can be
              recommended according to prakriti to restore this balance and
              provide health.
            </p>
            {/* More content... */}
            <div className="mt-4">
              <h1 className="text-2xl font-bold underline">
                Importance of Knowing one's Prakriti :
              </h1>
              <p className="text-justify">
                According to Ayurvedic texts prakriti tells us about the
                susceptibility of an individual to develop particular type of
                diseases throughout his lifetime. There is a beautiful example
                that if a vata type person develops vataj disorder then its
                prognosis is difficult. If kapha type or pitta type person
                develops similar vataj disease then the prognosis is better and
                disease is likely to be cured easily. Prakriti Analysis by means
                of Tridoshas or Trienergies of body plays a very important role
                in diagnosis and treatment of the disease. Not only this but
                this Diagnostic tool also helps you to know about particular
                dietary regimens, herbs to avoid or to prefer. Prakriti Analysis
                helps us to maintain healthy life style as well.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
