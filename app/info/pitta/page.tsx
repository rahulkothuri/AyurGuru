import Image from "next/image";
import banner from "@/public/info/pittha.png";
export default function Pitta() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen">
      <div className="max-w-4xl p-4">
        <Image src={banner} alt="Banner" width={1200} height={600} />
        <div className="mt-4">
          <h1 className="text-5xl font-bold">Pitta</h1>
          <p className="mt-2 text-gray-600">Published on January 21, 2024</p>
          <div className="mt-4 space-y-7">
            {/* Your blog post content goes here */}
            <p className="text-justify ">
              Agni or fire is the symbol of Pitta. The pivotal function of Pitta
              is transformation. It takes care of the metabolic activities of
              the body, digestion and release of energy. This component of tri
              energy governs the transformation of body and mind. It helps to
              discriminate between right and wrong and it helps to visualize how
              we perceive things through our senses. Individuals with Pitta
              characteristics are generally fiery and aggressive in nature. They
              manifest a sense of anger in mind and body as well.
            </p>
            {/* More content... */}
            <div className="mt-4">
              <h1 className="text-2xl font-bold underline">
                Recommended Diet to be Followed for Pacifying Pitta :
              </h1>
              <p className="text-justify">
                <p>
                  1. Consume a lot of dairy products, which include ghee, milk,
                  and butter. Ghee is highly recommended for individuals with
                  Pitta characteristics. Individuals with Pitta attributes must
                  not consume yogurt, cheese, sour cream.
                  <br />
                  2. Oils like sunflower, coconut, and olive oils are the best
                  ones for pacifying Pitta. Consumption of almond, corn, and
                  sesame oil must be completely avoided for people with Pitta
                  characteristics.
                  <br />
                  3. Consumption of sweeteners is recommended except that of
                  honey and molasses.
                  <br />
                  4. Some best grains to be consumed are barley, oats, rice, and
                  wheat. However, corn, millet, brown rice, and rye consumption
                  must generally be avoided.
                  <br />
                  5. Reduction in the intake of the sour fruits like apricots,
                  berries, and grapefruits and an increase in the intake of the
                  sweeter fruits like melons, cherries, avocados, coconuts,
                  mangoes, melons, grapes, plums ripe pineapples is recommended.
                  <br />
                  6. Asparagus, potatoes, cucumbers, sweet potatoes, and green
                  leafy vegetables are generally suggested. Also, lettuce,
                  pumpkins, broccoli, cauliflower, okra, celery, green beans,
                  and lettuce are some of the vegetables, which are recommended
                  for individuals with Pitta characteristics. And, the
                  individual must not consume hot peppers, tomatoes, beets,
                  carrots, onion, radish, garlic, spinach, and eggplant at all.
                  <br />
                  7. People, who are non-vegetarians must consume turkey,
                  pheasant, and chicken. Nevertheless, consumption of seafood,
                  beef, and eggs must be completely minimized.
                  <br />
                  8. Coriander, cardamom, cilantro, fennel, and saffron are some
                  of the spices, which impart cooling and soothing nature to the
                  people. These must be consumed. While the hotter spices like
                  cumin, ginger, black pepper, clove, fenugreek, salt, cloves,
                  mustard, cayenne, and chili peppers must be completely avoided
                  since they instigate the fiery nature further.
                  <br />
                  Pitta dosha heats up the body and mind. So, consume cool
                  liquids and foods. Foods, which are bitter, astringent, and
                  sweet in taste are the best, while foods, which are salty,
                  pungent, sour must be completely avoided.
                </p>
              </p>
            </div>
            <div className="mt-4">
              <h1 className="text-xl font-bold underline">
                Tips to Balance Pitta :
              </h1>
              <p className="text-justify">
                <p>
                  A slight modification in lifestyle will provide wonderful ways
                  to balance Pitta in individuals:
                  <br />
                  1. Choose the ones that provide soothing, sweet, stabilizing,
                  and cooling impacts to the individuals.
                  <br />
                  2. Take an adequate amount of rest.
                  <br />
                  3. Also, practice some physical activities as well.
                  <br />
                  4. Do not skip meals. Eat foods that are bitter, astringent,
                  and sweet. Also, consume foods that induce coolness. These
                  include melons, cucumbers, and sweet fruits.
                  <br />
                  5. It is recommended to spend some time with nature on a daily
                  basis. Take a stroll amidst the wood and water. Keep fresh
                  plants and flowers. Also, if possible, walk in the moonlight.
                  <br />
                  6. Go for an oil massage daily with coconut or olive or any
                  other cooler oils.
                  <br />
                  7. Opt for cool colors, which include green, silver, and blue.
                  <br />
                  8. Always keep laughing. It is a great healer.
                  <br />
                  9. Go for aromas, which are sweet and cool. These include
                  rose, sandalwood, mint, jasmine, fennel, chamomile, and
                  lavender.
                </p>
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
