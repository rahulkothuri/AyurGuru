import Image from "next/image";
import banner from "@/public/info/kapha.png";
export default function Kapha() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen">
      <div className="max-w-4xl p-4">
        <Image src={banner} alt="Banner" width={1200} height={600} />
        <div className="mt-4">
          <h1 className="text-5xl font-bold">Kapha</h1>
          <p className="mt-2 text-gray-600">Published on January 21, 2024</p>
          <div className="mt-4 space-y-7">
            {/* Your blog post content goes here */}
            <p className="text-justify ">
              Ayurveda, the ancient medical science focuses on three major
              energies, which are known as the tri-energies. One of them is
              Kapha. Kapha mainly decides the body structure of the individuals.
              It is that energy, which takes hold of all the cells as a mass and
              helps to form the spines, bones, muscles and fats. The major
              function of this element is protection of the body. It takes care
              of lubrication of the human structure both physically and
              psychologically as well. It also helps in the formation of the 7
              major body tissues of the human beings, namely, blood, nutritive
              fluids, fat, bones, marrow, reproductive tissues and muscles. It
              also helps in controlling the growth, lubrication and weight of
              the lungs and joints.
            </p>
            {/* More content... */}
            <div className="mt-4">
              <h1 className="text-2xl font-bold underline">
                Recommended Diet to be Followed for Pacifying kapha :
              </h1>
              <p className="text-justify">
                Ayurveda states that every individual must concentrate on the
                diet and food items to be consumed to balance the impact of
                doshas or pacifying it from becoming aggravated. Kapha generally
                is cold, heavy, and oily. So, light, warm, and dry food items
                are generally preferred. Also, foods bitter, pungent, and
                astringent in taste are most effective. Avoid foods with sour,
                salty, and sweet in taste.
                <br />
                <br />
                1. Reduce the consumption of dairy products, which enhance the
                aggravating properties of Kapha. Very less quantity of low-fat
                yogurt, milk, and ghee is recommended.
                <br />
                2. Go for a liquid diet once weekly. Consume juices of fresh
                fruits and vegetables and pureed soup.
                <br />
                3. Consume honey since it acts wonderfully to pacify Kapha.
                Avoid other sweeteners, which enhance the Kapha doshas. This
                further leads to problems like allergies, cold, lethargy, and
                sinus.
                <br />
                4. Apples, cranberries, pomegranates, apricots, and other
                lighter fruits are generally recommended. While, pineapples,
                avocados, peaches, oranges, coconuts, dates, melons, figs, and
                other heavier fruits must be avoided.
                <br />
                5. Chicken, eggs, turkey, and seafood are suggested for
                individuals to maintain a balance for those with Kapha traits.
                Red meat shall be avoided completely to maintain the balance.
                <br />
                6. Pungent spices like cayenne, pepper, and ginger can be
                included in the diet. Salt must be avoided completely.
                <br />
                7. Individuals can apply a very little amount of extra virgin
                almond oil, ghee, olive oil, sunflower oil, safflower oil, and
                mustard oil.
                <br />
                8. People with Kapha characteristics favor corn, barley, rye,
                buckwheat, and millet. They must decrease the consumption of
                wheat, oats, and rice.
                <br />
                9. All vegetables except the ones, which are juicy and sweet
                like zucchini, tomatoes, sweet potatoes must be consumed.
                <br />
                10. Reduce the intake of all nuts.
                <br />
                11. Consume ginger tea, which has a stimulating action on the
                process of digestion and the whole metabolic process of the
                body.
              </p>
            </div>
            <div className="mt-4">
              <h1 className="text-xl font-bold underline">
              Tips to Balance Kapha :
              </h1>
              <p className="text-justify">
              <p>
      1. Keep yourselves warm and get away from dampness. Individuals are highly sensitive to cold and get benefit from heat.
      <br />
      2. Follow a regular routine. Do not take naps in between in the morning.
      <br />
      3. Perform a daily dry massage to maintain a balanced process of circulation in the body.
      <br />
      4. Cleanliness is another important suggestion for individuals with Kapha characteristics. Keep home, car, and office clean and dispose of all those things, which are not required at all.
      <br />
      5. Carry out exercises regularly. This helps to prevent the accumulation of harmful toxins from the body. Practice swimming, running, aerobics, and bicycling. Dance to the rhythm of music.
      <br />
      6. Consume spices, which have stimulating aromas like cinnamon, cloves, juniper, and camphor.
      <br />
      7. Make use of bright and warm colors like red, yellow, and orange.
    </p>
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
