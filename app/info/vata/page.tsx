import Image from "next/image";
import banner from "@/public/info/vata.png";
export default function Vata() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen">
      <div className="max-w-4xl p-4">
        <Image src={banner} alt="Banner" width={1200} height={600} />
        <div className="mt-4">
          <h1 className="text-5xl font-bold">Vata</h1>
          <p className="mt-2 text-gray-600">Published on January 21, 2024</p>
          <div className="mt-4 space-y-7">
            {/* Your blog post content goes here */}
            <p className="text-justify ">
              The major concept in Ayurveda is the concept of tri-energies. This
              concept revolves around the three major components of the world -
              Vata, Pitta and Kapha. Of this, Vata symbolizes motion. It depicts
              movement of the body and the physiological process of elimination
              or expelling the waste materials out from the body. It also has a
              role to play in maintaining the functioning of the nervous system.
              Most importantly, Vata has influencing action on other doshas as
              well. It is considered to be the leader of all the three
              principles. This is because; Kapha and Pitta are unable to move
              without Vata at all Vata is the main governing energy, which
              focuses on the movement of body as well as mind.
            </p>
            {/* More content... */}
            <div className="mt-4">
              <h1 className="text-2xl font-bold underline">
                Recommended Diet to be Followed for Pacifying Vata :
              </h1>
              <p className="text-justify">
                <p>
                  1. Dairy products - Any dairy product pacifies Vata. Boil milk
                  and drink the warm milk. However, do not consume milk after
                  having a full meal.
                  <br />
                  2. Consume a large quantity of food. But, maintain the limit.
                  It must not exceed such that you fail to digest it easily.
                  <br />
                  3. Sweeteners
                  <br />
                  4. Sweet and sour fruits like bananas, oranges, grapes,
                  avocados, peaches, cherries, melons, plums, berries,
                  pineapples, papayas, and mangoes. Limit the consumption of
                  light and dry fruits like pears, apples, cranberries, and
                  pomegranates.
                  <br />
                  5. Vegetables like cucumbers, beets, asparagus, sweet
                  potatoes, and carrots. But, do not eat raw. Always wash and
                  cook the vegetables properly. It is advised to reduce the
                  consumption of cabbage and sprouts.
                  <br />
                  6. Spices like cumin, cardamom, cinnamon, ginger, cloves,
                  salt, and a little amount of black pepper are recommended.
                  <br />
                  7. Nuts
                  <br />
                  8. Oils
                  <br />
                  9. Beans except mung dal, tofu
                  <br />
                  10. Reduce the intake of the food items, which are bitter,
                  astringent, or pungent in taste.
                </p>
              </p>
            </div>
            <div className="mt-4">
              <h1 className="text-xl font-bold underline">
              Balancing Vata - Healthy tips in Lifestyle :
              </h1>
              <p className="text-justify">
                <p>
                  Slight changes in lifestyle are required to maintain a balance
                  between the Vata characteristics properly. Some of them are
                  given below:
                  <br />
                  1. Abhyanga: Daily relaxing massage with Ayurvedic sesame oil
                  or almond oil
                  <br />
                  2. Maintaining a warm temperature
                  <br />
                  3. Early to bed
                  <br />
                  4. Sufficient rest
                  <br />
                  5. Consumption of warm and oily food
                  <br />
                  6. Daily elimination
                  <br />
                  7. Lot of rest
                  <br />
                  8. Drink seasonal meals and organic herbal tea
                  <br />
                  9. Avoiding stimulants completely
                  <br />
                  10. Eat warm, heavy, and oily food
                  <br />
                  11. Maintain a regular routine
                  <br />
                  12. Do not skip between meals
                  <br />
                  13. Maintain the warmth of the body. Avoid being chilled.
                  Enjoy music, which imparts soothing and calming impact on the
                  body
                  <br />
                  14. Yoga, walking, cycling, golf, light tennis, aerobics, and
                  other light exercises must be practiced on a daily basis.
                  <br />
                  15. Wear warm-colored clothes like pastels, brown, earth
                  colors, and yellows.
                  <br />
                  16. Favor sweet, warm, and heavy aromas, which include bay,
                  basil, lavender, pine, vanilla, citrus, cloves, cinnamon,
                  sage.
                  <br />
                  It is important to maintain stability, warmth, and consistency
                  in lives if you have the predominant characteristic of Vata.
                </p>
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
