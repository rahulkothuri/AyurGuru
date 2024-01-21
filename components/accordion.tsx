// Import necessary modules
import { useState, useEffect } from 'react';

// Accordion component
const Accordion = () => {
  const [openIndex, setOpenIndex] = useState<number | null>(null);

  useEffect(() => {
    // Close all accordions when the component mounts
    closeAllAccordions();
  }, []);

  const handleAccordionClick = (index: number) => {
    setOpenIndex((prevIndex) => (prevIndex === index ? null : index));
  };

  const closeAllAccordions = () => {
    setOpenIndex(null);
  };

  return (
    <div className='pt-7 space-y-4'>
      {[0, 1, 2].map((index) => (
        <div className="collapse collapse-plus bg-gray-100" key={index}>
          <input
            type="radio"
            name="my-accordion-3"
            checked={openIndex === index}
            onChange={() => handleAccordionClick(index)}
          />
          <div className="collapse-title text-xl font-medium">
          Why prakrithi matters?
          </div>
          <div
            className="collapse-content"
            style={{ maxHeight: openIndex === index ? '1000px' : '0' }}
          >
            <p>hello</p>
          </div>
        </div>
      ))}
    </div>
  );
};

export default Accordion;
